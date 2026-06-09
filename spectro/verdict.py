"""Evidence-score verdict logic.

Computes per-feature subscores (0-100) and combines into a lossy_evidence_score
plus an orthogonal data_quality_score. Verdict comes from both.

References:
- LAME cutoff tables (timothygu.me/lame) for codec anchors
- arxiv 2407.21545 for ROC-curve framing
- ITU-R BS.1387 (PEAQ) for multi-feature fusion via floor-bounded sum
"""
from typing import Dict, Tuple

import numpy as np

from .dataclasses_ import EvidenceFlag, T

# I'll leave comments in here because it can help if anyone wants to audit it, please keep in mind I'm still learning this so rookie mistakes WILL ABSOLUTELY be made!

def _score_edge_p90(edge_p90: float, nyquist: float) -> float:
    """Distance of p90 edge from nearest codec cutoff anchor."""
    if edge_p90 >= 21000:
        return 0.0  # near Nyquist for 44.1k = no evidence
    if edge_p90 >= nyquist * 0.97:
        return 0.0  # full-band, no evidence
    dists = [abs(edge_p90 - c) for c in T.codec_cutoffs]
    nearest = min(dists)
    if nearest <= T.codec_cutoff_window_hz:
        # A tight match → high score
        return 100.0 * (1.0 - nearest / T.codec_cutoff_window_hz)
    # Lower frequencies can be more suspicious (he-aac, low-bitrate, etc)
    if edge_p90 < 14000:
        return 80.0
    if edge_p90 < 17000:
        return 50.0
    if edge_p90 < 19500:
        return 30.0
    return 10.0


def _score_slope(slope_db_per_khz: float) -> float:
    """Steeper transition band = more lossy."""
    s = abs(slope_db_per_khz)
    if s < 5:
        return 0.0   # natural rolloff
    if s >= 40:
        return 100.0  # brick-wall (like opus)
    # Linear ramp 5..40 → 0..100, with knee at lossy threshold
    if s >= abs(T.slope_strong_threshold):
        return 80.0 + (s - abs(T.slope_strong_threshold)) * 2.0
    if s >= abs(T.slope_lossy_threshold):
        return 40.0 + (s - abs(T.slope_lossy_threshold)) * (40.0 / 15.0)
    return s * (40.0 / abs(T.slope_lossy_threshold))


def _score_jitter(jitter_hz: float) -> float:
    """Low jitter = codec locks edge = lossy."""
    if jitter_hz >= 800:
        return 0.0
    if jitter_hz <= T.jitter_strong_threshold:
        return 90.0
    if jitter_hz <= T.jitter_lossy_threshold:
        return 60.0
    if jitter_hz <= 500:
        return 30.0
    return 10.0


def _score_high_band(band_ratio_db: float) -> float:
    """Severely suppressed high-band = lossy or analog."""
    if band_ratio_db >= -10:
        return 0.0
    if band_ratio_db <= T.band_ratio_strong_dB:
        return 90.0
    if band_ratio_db <= T.band_ratio_lossy_dB:
        return 60.0
    # -20 to -10 dB range
    return 30.0 * ((-10 - band_ratio_db) / 10.0)


def _score_rolloff_var(rolloff_var_hz: float) -> float:
    """Low variance = locked = lossy."""
    if rolloff_var_hz >= 1000:
        return 0.0
    if rolloff_var_hz <= 150:
        return 80.0
    if rolloff_var_hz <= T.rolloff_var_lossy_threshold:
        return 50.0
    return max(0.0, 50.0 * (1.0 - (rolloff_var_hz - T.rolloff_var_lossy_threshold) / 700.0))


def _score_sbr(sbr_likelihood: str) -> float:
    if sbr_likelihood == "likely":
        return 85.0
    if sbr_likelihood == "possible":
        return 40.0
    return 0.0


def _score_persistence(persistence: float, hard_cutoff: bool) -> float:
    """Backstop: existing cutoff_persistence signal."""
    if not hard_cutoff:
        return 0.0
    if persistence >= 0.7:
        return 100.0
    if persistence >= 0.5:
        return 70.0
    if persistence >= 0.3:
        return 40.0
    return 15.0


def compute_lossy_score(
    edge_p90: float, slope_db_per_khz: float, jitter_hz: float,
    band_ratio_db: float, rolloff_var_hz: float, sbr_likelihood: str,
    persistence: float, hard_cutoff: bool, nyquist: float,
) -> Tuple[float, Dict[str, float]]:
    """Weighted sum of subscores. Returns (final_0_100, subscores dict)."""
    subs = {
        "edge":        _score_edge_p90(edge_p90, nyquist),
        "slope":       _score_slope(slope_db_per_khz),
        "jitter":      _score_jitter(jitter_hz),
        "high_band":   _score_high_band(band_ratio_db),
        "rolloff":     _score_rolloff_var(rolloff_var_hz),
        "sbr":         _score_sbr(sbr_likelihood),
        "persistence": _score_persistence(persistence, hard_cutoff),
    }
    # jitter only counts when the edge sits somewhere codec-like, analog masters can hold a dead-stable natural rolloff
    if subs["edge"] == 0.0 and subs["slope"] == 0.0:
        subs["jitter"] = min(subs["jitter"], 25.0)
    weights = {
        "edge":        T.w_edge,
        "slope":       T.w_slope,
        "jitter":      T.w_jitter,
        "high_band":   T.w_high_band,
        "rolloff":     T.w_rolloff,
        "sbr":         T.w_sbr,
        "persistence": T.w_persistence,
    }
    total = sum(weights.values())
    score = sum(subs[k] * weights[k] for k in subs) / total
    # brick wall at a codec anchor that never moves = the classic signature
    if subs["edge"] >= 80 and subs["slope"] >= 80 and subs["persistence"] >= 70:
        score = max(score, 75.0)
    return min(100.0, max(0.0, score)), subs


def compute_quality_score(
    rms_db: float, edge_p90: float, edge_p50: float, noise_floor: float,
    sr: int,
) -> Tuple[float, Dict[str, float]]:
    """Orthogonal: 'is the data trustworthy enough to decide?'."""
    penalty = 0.0
    breakdown = {}
    # Silent or very quiet → can't analyze
    if rms_db < T.quality_silent_rms_dB:
        p = 40.0
        penalty += p
        breakdown["silent"] = p
    elif rms_db < -45:
        p = 15.0
        penalty += p
        breakdown["quiet"] = p
    # Extremely narrow bandwidth (telephone, AM radio)
    if edge_p90 < T.quality_narrow_bandwidth_hz:
        p = 35.0
        penalty += p
        breakdown["narrow_band"] = p
    elif edge_p50 < 4000:
        p = 20.0
        penalty += p
        breakdown["very_narrow_median"] = p
    # Elevated noise floor (vinyl, tape, saturation)
    if noise_floor > T.quality_high_noise_floor_dB:
        p = 15.0
        penalty += p
        breakdown["high_noise_floor"] = p
    score = max(0.0, 100.0 - penalty)
    return score, breakdown


def decide_verdict(lossy_score: float, quality_score: float) -> Tuple[str, str]:
    """Map (lossy, quality) → (verdict, explanation)."""
    if quality_score < T.quality_required:
        return "INCONCLUSIVE", "not enough usable signal to judge, could be silent, narrowband or analog"
    if lossy_score >= T.fail_score:
        return "FAIL", "this looks transcoded"
    if lossy_score >= T.warn_score:
        return "WARN", "some lossy traits here, possible transcode or just heavy-handed mastering"
    return "PASS", "no lossy fingerprints found"


def build_score_flags(subscores: Dict[str, float], lossy_score: float,
                      quality_breakdown: Dict[str, float]) -> list:
    """Surface dominant signals as EvidenceFlags so users see WHY."""
    flags = []

    def sev_from_sub(name: str, s: float) -> str:
        if s >= 80: return "high"
        if s >= 50: return "medium"
        if s >= 25: return "low"
        return "info"

    descriptors = {
        "edge":        ("Spectral edge", "active edge near known codec cutoff"),
        "slope":       ("Steep filter slope", "transition band falls faster than natural rolloff"),
        "jitter":      ("Locked spectral edge", "edge frequency barely varies across frames"),
        "high_band":   ("Suppressed high band", "energy above 4 kHz severely reduced"),
        "rolloff":     ("Stable rolloff", "85% energy point too consistent for natural audio"),
        "sbr":         ("SBR-like reconstruction", "upper band correlates with lower (HE-AAC pattern)"),
        "persistence": ("Persistent hard cutoff", "drop holds across active frames"),
    }
    for name, s in sorted(subscores.items(), key=lambda x: -x[1]):
        if s < 25:
            continue
        label, detail = descriptors[name]
        flags.append(EvidenceFlag(severity=sev_from_sub(name, s), name=label, detail=f"{detail} (subscore {s:.0f}/100)"))
    if "silent" in quality_breakdown or "narrow_band" in quality_breakdown:
        reasons = ", ".join(quality_breakdown.keys())
        flags.append(EvidenceFlag(severity="info", name="Low data quality",
                                  detail=f"verdict reliability reduced: {reasons}"))
    return flags


# - Feature Extraction Helpers -
def steepest_slope_db_per_khz(avg_db: np.ndarray, freqs: np.ndarray,
                              lo_hz: float, hi_hz: float, span_hz: float = 250.0) -> float:
    """Steepest drop over a 250 Hz span. Codec shelves are near-vertical,
    a symmetric regression window dilutes them with passband."""
    from scipy.ndimage import gaussian_filter1d
    bin_hz = freqs[1] - freqs[0]
    k = max(1, int(span_hz / bin_hz))
    i0 = np.searchsorted(freqs, lo_hz)
    i1 = np.searchsorted(freqs, hi_hz)
    if i1 - i0 < k + 4:
        return 0.0
    smoothed = gaussian_filter1d(avg_db, sigma=3)
    diffs = (smoothed[i0 + k:i1] - smoothed[i0:i1 - k]) / (k * bin_hz) * 1000.0
    return float(np.min(diffs))


def compute_slope_db_per_khz(avg_db: np.ndarray, freqs: np.ndarray,
                              center_hz: float, window_hz: float = None) -> float:
    """Linear regression slope (dB/kHz) over transition band centered on center_hz."""
    if window_hz is None:
        window_hz = T.slope_window_hz
    half = window_hz / 2
    mask = (freqs >= center_hz - half) & (freqs <= center_hz + half)
    if np.sum(mask) < 4:
        return 0.0
    f_kHz = freqs[mask] / 1000.0
    y = avg_db[mask]
    # least-squares slope
    A = np.vstack([f_kHz, np.ones_like(f_kHz)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def compute_edge_jitter(edges_all: np.ndarray, active_mask: np.ndarray) -> float:
    """MAD of active edge over time, in Hz. Only over active frames."""
    valid = ~np.isnan(edges_all)
    if active_mask is not None and len(active_mask) == len(edges_all):
        valid = valid & active_mask
    edges = edges_all[valid]
    if len(edges) < 8:
        return 0.0
    med = np.median(edges)
    mad = float(np.median(np.abs(edges - med)))
    return mad


def compute_rolloff_85_variance(Sxx_db: np.ndarray, freqs: np.ndarray) -> float:
    """Std of the 85th-percentile energy frequency, per frame, over time."""
    if Sxx_db.shape[1] < 8:
        return 0.0
    S_lin = 10 ** (Sxx_db / 10)
    total_per_frame = np.sum(S_lin, axis=0) + 1e-12
    cum = np.cumsum(S_lin, axis=0)
    norm = cum / total_per_frame[np.newaxis, :]
    # For each frame, first bin where cumulative reaches 0.85
    idx = np.argmax(norm >= 0.85, axis=0)
    rolloffs = freqs[np.minimum(idx, len(freqs) - 1)]
    return float(np.std(rolloffs))


def compute_band_ratio_db(avg_db: np.ndarray, freqs: np.ndarray,
                           split_hz: float = 4000.0) -> float:
    """10*log10(E_high / E_low) using split at split_hz."""
    avg_lin = 10 ** (avg_db / 10)
    low_mask = freqs < split_hz
    high_mask = freqs >= split_hz
    e_low = np.sum(avg_lin[low_mask]) + 1e-12
    e_high = np.sum(avg_lin[high_mask]) + 1e-12
    return float(10 * np.log10(e_high / e_low))
