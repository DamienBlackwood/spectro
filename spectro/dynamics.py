import numpy as np

from .dataclasses_ import DR_THRESHOLDS, DynamicsResult, T


def clip_runs(peaks: np.ndarray, sr: int, limit: int = 8) -> list:
    """Start times of the first few clipped stretches, not every sample."""
    hits = np.flatnonzero(peaks >= T.clip_threshold)
    if len(hits) == 0:
        return []
    # a run is a group of hits with no real gap between them
    gap = max(1, int(0.01 * sr))
    starts = hits[np.r_[True, np.diff(hits) > gap]]
    return (starts[:limit] / sr).tolist()


def analyze_dynamics(data: np.ndarray, sr: int) -> DynamicsResult:
    """Analyze dynamic range, clipping, and crest factor."""
    peak = np.max(np.abs(data))
    peak_db = 20 * np.log10(peak + 1e-10)

    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    crest_factor = peak_db - rms_db

    clipped_samples = np.sum(np.abs(data) >= T.clip_threshold)
    clip_percentage = (clipped_samples / data.size) * 100

    # per channel. a downmix cancels out-of-phase clipping and hides it
    peaks = np.max(np.abs(data), axis=1) if data.ndim > 1 else np.abs(data)
    clip_times = clip_runs(peaks, sr)

    mono_data = data.mean(axis=1) if data.ndim > 1 else data

    window_size = int(T.clip_window_sec * sr)
    n_windows = len(mono_data) // window_size
    if n_windows > 0:
        windowed = mono_data[:n_windows * window_size].reshape(n_windows, window_size)
        window_rms = np.sqrt(np.mean(windowed ** 2, axis=1))
        window_rms_db = 20 * np.log10(window_rms + 1e-10)

        loud = np.percentile(window_rms_db, 95)
        quiet = np.percentile(window_rms_db, 5)
        dynamic_range = loud - quiet
    else:
        dynamic_range = 0

    dr_rating = "dynamic"
    if crest_factor < DR_THRESHOLDS['brickwalled']:
        dr_rating = "brickwalled"
    elif crest_factor < DR_THRESHOLDS['compressed']:
        dr_rating = "compressed"
    elif crest_factor < DR_THRESHOLDS['moderate']:
        dr_rating = "moderate"

    return DynamicsResult(
        peak_db=peak_db, rms_db=rms_db, crest_factor=crest_factor,
        dynamic_range=dynamic_range, dr_rating=dr_rating,
        clipped_samples=clipped_samples, clip_percentage=clip_percentage,
        clip_times=clip_times
    )
