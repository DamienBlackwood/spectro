from dataclasses import asdict
from typing import Optional

from .dataclasses_ import SpectralAnalysisResult


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.2f}s"
    return f"{int(s//60)}m {s%60:.1f}s"


def print_limitations():
    print("\n  Keep in mind:")
    print("    • this can't PROVE a file is lossless")
    print("    • high-bitrate MP3/AAC/Opus can look squeaky clean here")
    print("    • plenty of genuine masters just don't have much top end")
    print("    • real proof needs a trusted source, not a spectrogram")


def build_json_report(res: SpectralAnalysisResult, container_codec: str, container_sr: str,
                      container_bitrate: Optional[str], container_bit_depth: Optional[str] = None) -> dict:
    return {
        "verdict": res.evidence.verdict,
        "explanation": res.evidence.explanation,
        "container_codec": container_codec,
        "container_sample_rate_hz": int(container_sr) if container_sr.isdigit() else None,
        "container_bitrate": container_bitrate,
        "container_bit_depth": container_bit_depth,
        "nyquist_hz": res.nyquist,
        "cutoff_hz": res.evidence.cutoff_freq,
        "shelf_type": res.evidence.shelf_type,
        "hard_cutoff": res.evidence.hard_cutoff,
        "strongest_drop_hz": res.evidence.best_drop_freq,
        "strongest_drop_db": res.evidence.max_drop_db,
        "active_edge_p10_hz": res.evidence.edge_p10,
        "active_edge_p50_hz": res.evidence.edge_p50,
        "active_edge_p90_hz": res.evidence.edge_p90,
        "active_edge_p97_hz": res.evidence.edge_p97,
        "cutoff_persistence": res.evidence.cutoff_persistence,
        "active_frames_pct": res.evidence.active_frames_pct,
        "high_band_db": res.evidence.high_band_db,
        "near_nyquist_db": res.evidence.near_nyquist_db,
        "sbr_likelihood": res.evidence.sbr_likelihood,
        "ultrasonic_delta_db": res.ultrasonic_delta,
        "transcode_warning": res.transcode_warning,
        "suspicious_windows": res.evidence.suspicious_windows,
        "closest_resemblance": res.profile,
        "lossy_score": res.evidence.lossy_score,
        "quality_score": res.evidence.quality_score,
        "subscores": res.evidence.subscores,
        "edge_jitter_hz": res.evidence.edge_jitter_hz,
        "rolloff_85_var_hz": res.evidence.rolloff_85_var_hz,
        "max_slope_db_per_khz": res.evidence.max_slope_db_per_khz,
        "band_ratio_db": res.evidence.band_ratio_db,
        "flags": [asdict(f) for f in res.evidence.suspicious_flags],
    }
