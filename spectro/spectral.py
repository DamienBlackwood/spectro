from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft

from . import verdict as verdict_mod
from .dataclasses_ import (
    CODEC_PROFILES, ChannelAnalysis, EvidenceFlag, SpectralAnalysisResult, T,
    TranscodeEvidence,
)


def stft_params(n_samples: int, nperseg: int) -> Tuple[int, int]:
    """Shrink the window to fit short files. scipy clamps nperseg on its own and
    then trips over the noverlap it was handed."""
    if n_samples < 1:
        raise ValueError("audio has no samples")
    nperseg = min(nperseg, n_samples)
    return nperseg, nperseg // 2


def active_band_edge(S_db: np.ndarray, freqs: np.ndarray,
                     floor_margin_db: float = T.floor_margin_db) -> np.ndarray:
    """Estimate highest frequency with meaningful energy per frame. Vectorized."""
    noise_floors = np.percentile(S_db, 10, axis=0)
    active = S_db > (noise_floors + floor_margin_db)
    any_active = active.any(axis=0)
    highest_idx = active.shape[0] - 1 - active[::-1, :].argmax(axis=0)
    edges = np.where(any_active, freqs[highest_idx], np.nan)
    return edges


def band_mean(avg_db: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.nan
    return float(np.mean(avg_db[mask]))


def cutoff_drop_score(avg_db: np.ndarray, freqs: np.ndarray, cutoff_hz: float,
                      width: float = T.cutoff_drop_width_hz) -> float:
    below = band_mean(avg_db, freqs, cutoff_hz - width, cutoff_hz)
    above = band_mean(avg_db, freqs, cutoff_hz, cutoff_hz + width)
    if np.isnan(below) or np.isnan(above):
        return 0.0
    return below - above


def strongest_drop(avg_db: np.ndarray, freqs: np.ndarray,
                   nyquist: float) -> Tuple[float, float]:
    """Sweep the candidate cutoffs and return the deepest (freq_hz, depth_db)."""
    cands = [c for c in T.candidate_cutoffs if c < nyquist * T.nyquist_cutoff_cand_cap]
    if not cands:
        return float(nyquist), 0.0
    drops = [(cutoff_drop_score(avg_db, freqs, c), -c) for c in cands]
    depth, neg_freq = max(drops)
    return float(-neg_freq), float(depth)


def get_active_frames(Sxx_db: np.ndarray, freqs: np.ndarray, noise_floor: float,
                      nyquist: float) -> np.ndarray:
    """Boolean mask of frames that are loud enough and spectrally rich."""
    frame_mean = np.mean(Sxx_db, axis=0)
    high_mask = (freqs >= T.high_band_low_hz) & (freqs < min(T.high_band_high_hz, nyquist * T.nyquist_high_band_cap))
    if not np.any(high_mask):
        high_band_energy = np.full(Sxx_db.shape[1], noise_floor - 100)
    else:
        high_band_energy = np.mean(Sxx_db[high_mask, :], axis=0)
    active = (frame_mean > noise_floor + T.frame_mean_margin) & (high_band_energy > noise_floor + T.high_band_margin)
    return active


def spectral_flatness(power_band: np.ndarray) -> float:
    geometric = np.exp(np.mean(np.log(power_band + 1e-12)))
    arithmetic = np.mean(power_band + 1e-12)
    return geometric / arithmetic


def detect_sbr(avg_db: np.ndarray, freqs: np.ndarray, cutoff_idx: int,
               noise_floor: float) -> str:
    """HE-AAC rebuilds the top octave from the one under it, so the two bands
    correlate and the upper one comes out flatter."""
    below_start = max(0, cutoff_idx - 80)
    below_end = cutoff_idx - 20
    above_start = cutoff_idx + 20
    above_end = min(len(avg_db), cutoff_idx + 80)
    if below_end <= below_start or above_end <= above_start:
        return "none"

    below_region = avg_db[below_start:below_end]
    above_region = avg_db[above_start:above_end]
    below_energy = float(np.mean(below_region))
    above_energy = float(np.mean(above_region))

    if above_energy <= noise_floor + T.sbr_above_energy_margin:
        return "none"
    if (below_energy - above_energy) >= T.sbr_band_delta_db:
        return "none"

    min_len = min(len(below_region), len(above_region))
    if min_len <= 10:
        return "none"
    corr = np.corrcoef(below_region[:min_len], above_region[:min_len])[0, 1]
    if np.isnan(corr) or corr <= T.sbr_corr_threshold:
        return "none"

    upper_flatness = spectral_flatness(10 ** (above_region / 10))
    lower_flatness = spectral_flatness(10 ** (below_region / 10))
    if upper_flatness > lower_flatness * T.sbr_flatness_ratio:
        return "likely"
    return "possible"


def _analyze_channel(data_ch: np.ndarray, sr: int) -> ChannelAnalysis:
    """STFT one channel and locate its cutoff."""
    nperseg, noverlap = stft_params(len(data_ch), T.detect_nperseg)

    frequencies, times, Zxx = stft(data_ch, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    power = np.abs(Zxx) ** 2
    avg_spectrum = np.mean(power, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)

    time_decim = max(1, Zxx.shape[1] // 2000)
    Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decim]) ** 2 + 1e-10)
    times_decim = times[::time_decim]

    noise_floor = float(np.percentile(avg_db, 5))
    nyquist = sr / 2

    # sweeping for the deepest shelf. the old gradient walk latched onto the first dip past 12k and called it the cutoff, even on clean files
    cutoff_freq, depth = strongest_drop(avg_db, frequencies, nyquist)
    slope_hi = min(cutoff_freq + T.cutoff_refine_window_hz, nyquist * 0.98)
    slope = verdict_mod.steepest_slope_db_per_khz(
        avg_db, frequencies, T.cutoff_search_start_hz, slope_hi)
    shelf_type = verdict_mod.shelf_type_from_slope(slope, depth)

    sbr_likelihood = "none"
    if shelf_type != 'none' and cutoff_freq < T.sbr_max_cutoff_hz:
        cutoff_idx = int(np.searchsorted(frequencies, cutoff_freq))
        if cutoff_idx < len(frequencies) - 50:
            sbr_likelihood = detect_sbr(avg_db, frequencies, cutoff_idx, noise_floor)

    if shelf_type == 'none':
        cutoff_freq = float(nyquist)

    return ChannelAnalysis(
        cutoff_freq=cutoff_freq,
        shelf_type=shelf_type,
        shelf_depth_db=depth,
        slope_db_per_khz=slope,
        sbr_likelihood=sbr_likelihood,
        frequencies=frequencies,
        times=times_decim,
        Sxx_db=Sxx_db,
        avg_db=avg_db,
        Zxx_full=Zxx,
        times_full=times,
        hop=nperseg - noverlap,
    )


def analyze_time_windows(Zxx_full: np.ndarray, times_full: np.ndarray, hop: int,
                         sr: int, frequencies: np.ndarray, nyquist: float,
                         noise_floor: float) -> List[str]:
    """Slice existing STFT into ~5-second windows and scan for suspicious cutoffs."""
    frames_per_window = max(1, int(5 * sr / hop))
    n_frames = Zxx_full.shape[1]
    n_windows = max(1, n_frames // frames_per_window)
    suspicious = []

    search_start_idx = np.argmin(np.abs(frequencies - T.cutoff_search_start_hz))
    search_end_idx = np.searchsorted(frequencies, nyquist * T.nyquist_search_end)
    if search_end_idx <= search_start_idx:
        return suspicious

    power_full = np.abs(Zxx_full) ** 2

    for i in range(n_windows):
        f_start = i * frames_per_window
        f_end = min(f_start + frames_per_window, n_frames)
        if f_end - f_start < frames_per_window // 2:
            continue

        avg_spectrum = np.mean(power_full[:, f_start:f_end], axis=1)
        avg_db = 10 * np.log10(avg_spectrum + 1e-10)
        smoothed = gaussian_filter1d(avg_db, sigma=3)
        gradient = np.gradient(smoothed)

        search_gradient = gradient[search_start_idx:search_end_idx]
        search_spectrum = smoothed[search_start_idx:search_end_idx]
        baseline_grad = np.median(gradient[search_start_idx // 2:search_start_idx])
        threshold = baseline_grad - T.grad_threshold_offset

        for j in range(len(search_gradient)):
            if search_gradient[j] < threshold and search_spectrum[j] > noise_floor + T.drop_search_local_threshold:
                window_start = max(0, j - 5)
                window_end = min(len(search_gradient), j + 10)
                local_drop = np.min(search_gradient[window_start:window_end])
                if local_drop < T.timewin_local_drop:
                    cf = frequencies[search_start_idx + j]
                    if cf < nyquist * T.nyquist_suspicious_cap:
                        t0 = times_full[f_start]
                        t1 = times_full[f_end - 1]
                        suspicious.append(f"{t0:.0f}s–{t1:.0f}s  cutoff around {cf:.0f} Hz")
                    break

    return suspicious


def _build_evidence_shell(
    cutoff_freq: float, cutoff_persistence: float, hard_cutoff: bool,
    high_band_db: float, near_nyquist_db: float, noise_floor: float,
    sbr_likelihood: str,
) -> TranscodeEvidence:
    """Empty evidence container. Verdict + flags + scores filled in by caller."""
    return TranscodeEvidence(
        verdict="PASS",
        explanation="",
        cutoff_freq=cutoff_freq,
        cutoff_persistence=cutoff_persistence,
        hard_cutoff=hard_cutoff,
        high_band_db=high_band_db,
        near_nyquist_db=near_nyquist_db,
        noise_floor=noise_floor,
        suspicious_flags=[],
        edge_p10=0.0,
        edge_p50=0.0,
        edge_p90=0.0,
        best_drop_freq=0.0,
        max_drop_db=0.0,
        sbr_likelihood=sbr_likelihood,
        active_frames_pct=0.0,
        suspicious_windows=[],
    )


def analyze_transcode_evidence(data: np.ndarray, sr: int) -> SpectralAnalysisResult:
    # codec lowpass is global, the middle 150s is plenty to find it
    max_n = int(T.detect_max_seconds * sr)
    t_offset = 0.0
    if len(data) > max_n:
        start = (len(data) - max_n) // 2
        data = data[start:start + max_n]
        t_offset = start / sr

    if data.ndim == 1:
        channels = [data]
    else:
        channels = [data[:, ch] for ch in range(data.shape[1])]

    channel_results = [_analyze_channel(ch, sr) for ch in channels]
    if t_offset:
        for r in channel_results:
            r.times = r.times + t_offset
            r.times_full = r.times_full + t_offset

    # worst channel = deepest shelf, lowest cutoff. joint stereo can leave one side more messed up than the other
    channel_results.sort(key=lambda r: (-r.shelf_depth_db, r.cutoff_freq))
    worst_channel = channel_results[0]
    sbr_likelihood = worst_channel.sbr_likelihood
    frequencies = worst_channel.frequencies
    times_decim = worst_channel.times

    all_avg_db = np.array([r.avg_db for r in channel_results])
    avg_db = np.mean(all_avg_db, axis=0)
    Sxx_db = np.mean(np.array([r.Sxx_db for r in channel_results]), axis=0)

    noise_floor = np.percentile(avg_db, 5)
    nyquist = sr / 2

    edges_all = active_band_edge(Sxx_db, frequencies, floor_margin_db=T.floor_margin_db)

    active_mask = get_active_frames(Sxx_db, frequencies, noise_floor, nyquist)
    active_frames_pct = float(np.mean(active_mask)) if len(active_mask) > 0 else 0.0
    valid_edges = edges_all[~np.isnan(edges_all) & active_mask] if np.any(active_mask) else edges_all[~np.isnan(edges_all)]

    # percentiles over active frames only, silence drags the edge down
    finite_edges = edges_all[~np.isnan(edges_all)]
    edge_src = valid_edges if len(valid_edges) >= 8 else finite_edges
    if len(edge_src) > 0:
        edge_p10 = float(np.percentile(edge_src, 10))
        edge_p50 = float(np.percentile(edge_src, 50))
        edge_p90 = float(np.percentile(edge_src, 90))
        # p97 ~ codec ceiling. p90 tracks content, quiet songs sit way under the lowpass
        edge_p97 = float(np.percentile(edge_src, 97))
    else:
        edge_p10 = edge_p50 = edge_p90 = edge_p97 = 0.0

    high_band_db = band_mean(avg_db, frequencies, T.high_band_high_hz, min(20000, nyquist * T.nyquist_high_band_cap))
    near_nyquist_db = band_mean(avg_db, frequencies, nyquist * T.nyquist_near_low, nyquist * T.nyquist_near_high)
    if np.isnan(high_band_db):
        high_band_db = noise_floor
    if np.isnan(near_nyquist_db):
        near_nyquist_db = noise_floor

    # steepest drop up to just past the edge. codec shelves live below ~21k, anything steeper above that is anti-alias or natural content edge
    slope_hi = min(edge_p97 + 1500, 21000.0, nyquist * 0.95)
    max_slope = verdict_mod.steepest_slope_db_per_khz(avg_db, frequencies, T.cutoff_search_start_hz, slope_hi)

    best_drop_freq, max_drop = strongest_drop(avg_db, frequencies, nyquist)
    # a shallow drop still counts if the wall beside it is vertical, analog masters have little up there that a brick wall only costs 6-9 dB
    hard_cutoff = bool(
        max_drop > T.hard_cutoff_drop_db
        or (max_drop > T.soft_cutoff_drop_db
            and max_slope <= -T.hard_cutoff_slope_db_per_khz)
    )
    shelf_type = verdict_mod.shelf_type_from_slope(max_slope, max_drop)
    cutoff_freq = best_drop_freq if shelf_type != 'none' else float(nyquist)

    if hard_cutoff:
        persistence_limit = best_drop_freq + T.cutoff_persistence_pad_hz
    else:
        persistence_limit = nyquist * T.nyquist_persistence_default
    cutoff_persistence = float(np.mean(valid_edges < persistence_limit)) if len(valid_edges) > 0 else 0.0

    ultrasonic_energy = None
    ultrasonic_delta = None
    if sr > T.high_sample_rate_hz:
        idx_24k = np.searchsorted(frequencies, T.ultrasonic_floor_hz)
        if idx_24k < len(frequencies):
            ultrasonic_peak = float(np.max(avg_db[idx_24k:]))
            ultrasonic_energy = ultrasonic_peak
            ultrasonic_delta = ultrasonic_peak - noise_floor

    suspicious_windows = analyze_time_windows(
        worst_channel.Zxx_full, worst_channel.times_full, worst_channel.hop,
        sr, frequencies, nyquist, noise_floor,
    )

    edge_jitter = verdict_mod.compute_edge_jitter(edges_all, active_mask)
    rolloff_var = verdict_mod.compute_rolloff_85_variance(Sxx_db, frequencies)
    band_ratio = verdict_mod.compute_band_ratio_db(avg_db, frequencies)

    rms_overall = float(np.sqrt(np.mean(data ** 2))) if data.size > 0 else 0.0
    rms_db = 20 * np.log10(rms_overall + 1e-10)

    lossy_score, subscores = verdict_mod.compute_lossy_score(
        edge_p90=edge_p97, slope_db_per_khz=max_slope, jitter_hz=edge_jitter,
        band_ratio_db=band_ratio, rolloff_var_hz=rolloff_var,
        sbr_likelihood=sbr_likelihood, persistence=cutoff_persistence,
        hard_cutoff=hard_cutoff, nyquist=nyquist,
    )
    quality_score, q_breakdown = verdict_mod.compute_quality_score(
        rms_db=rms_db, edge_p90=edge_p90, edge_p50=edge_p50,
        noise_floor=noise_floor, sr=sr,
    )
    verdict_str, explanation = verdict_mod.decide_verdict(lossy_score, quality_score)

    evidence = _build_evidence_shell(
        cutoff_freq=cutoff_freq, cutoff_persistence=cutoff_persistence,
        hard_cutoff=hard_cutoff, high_band_db=high_band_db,
        near_nyquist_db=near_nyquist_db, noise_floor=noise_floor,
        sbr_likelihood=sbr_likelihood,
    )
    evidence.verdict = verdict_str
    evidence.explanation = explanation
    evidence.edge_p10 = edge_p10
    evidence.edge_p50 = edge_p50
    evidence.edge_p90 = edge_p90
    evidence.edge_p97 = edge_p97
    evidence.best_drop_freq = best_drop_freq
    evidence.max_drop_db = max_drop
    evidence.active_frames_pct = active_frames_pct
    evidence.suspicious_windows = suspicious_windows
    evidence.max_slope_db_per_khz = max_slope
    evidence.edge_jitter_hz = edge_jitter
    evidence.rolloff_85_var_hz = rolloff_var
    evidence.band_ratio_db = band_ratio
    evidence.lossy_score = lossy_score
    evidence.quality_score = quality_score
    evidence.subscores = subscores
    evidence.suspicious_flags = verdict_mod.build_score_flags(subscores, lossy_score, q_breakdown)

    scores = {}
    for codec, profile in CODEC_PROFILES.items():
        score = 0
        low, high = profile['cutoff']

        if low <= cutoff_freq <= high:
            score += 40
        elif abs(cutoff_freq - (low + high) / 2) < 2000:
            score += 20
        elif abs(cutoff_freq - (low + high) / 2) < 4000:
            score += 5

        if profile['shelf'] == shelf_type:
            score += 30
        elif profile['shelf'] in ['hard', 'medium'] and shelf_type in ['hard', 'medium']:
            score += 15
        elif profile['shelf'] in ['soft', 'none'] and shelf_type in ['soft', 'none']:
            score += 15

        if profile['sbr'] != (sbr_likelihood != "none"):
            score -= 40
        else:
            score += 30

        scores[codec] = score

    best_profile = max(scores, key=scores.get)

    transcode_warning = None
    if sr > T.high_sample_rate_hz and cutoff_freq < T.upsample_cutoff_hz and hard_cutoff:
        transcode_warning = f"{sr} Hz container but the cutoff sits at {cutoff_freq:.0f} Hz, smells like upsampled lossy"
    elif sr > T.high_sample_rate_hz and ultrasonic_delta is not None and ultrasonic_delta < T.ultrasonic_delta_threshold:
        transcode_warning = (
            f"{sr} Hz container but barely any ultrasonic content. Probably a 44.1/48 kHz source "
            "upsampled somewhere along the way, not necessarily lossy"
        )
    elif cutoff_freq < T.low_cutoff_hz and hard_cutoff and sbr_likelihood == "none":
        transcode_warning = f"cutoff down at {cutoff_freq:.0f} Hz points to a heavily compressed source"

    return SpectralAnalysisResult(
        profile=best_profile,
        cutoff_freq=cutoff_freq, shelf_type=shelf_type, sbr_likelihood=sbr_likelihood,
        transcode_warning=transcode_warning,
        ultrasonic_energy=ultrasonic_energy, ultrasonic_delta=ultrasonic_delta,
        noise_floor=noise_floor, scores=scores, frequencies=frequencies, times=times_decim,
        Sxx_db=Sxx_db, avg_spectrum_db=avg_db, nyquist=nyquist,
        evidence=evidence,
        edge_times=times_decim,
        edge_values=edges_all,
    )
