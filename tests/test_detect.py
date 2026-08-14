"""Synthetic checks, no audio files needed. Run with pytest or just run the file.

The real tuning happens against actual transcodes (see make_corpus.py), this is
here to catch the day someone refactors the scoring and quietly inverts it.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectro.dynamics import (                                  # noqa: E402
    DynamicsAccumulator, analyze_dynamics, clip_runs,
)
from spectro.commands.compare import _align_signals             # noqa: E402
from spectro.display import (                                   # noqa: E402
    analyze_display_array, compute_display_spectrogram,
)
from spectro.reporting import fmt_time                          # noqa: E402
from spectro.spectral import analyze_transcode_evidence         # noqa: E402
from spectro.term import _pool                                  # noqa: E402
from spectro.verdict import (                                   # noqa: E402
    _score_shelf_depth, _score_slope, compute_edge_jitter, shelf_type_from_slope,
)

SR = 44100
DUR = 20


def noise_track(cutoff=None, sr=SR, dur=DUR, seed=1):
    """Pink-ish noise with a natural taper, optionally brick-walled."""
    rng = np.random.default_rng(seed)
    n = sr * dur
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    shape = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
    shape *= np.exp(-freqs / (sr * 0.75))  # gentle top-end taper, no cliff
    spec *= shape
    if cutoff:
        spec[freqs > cutoff] = 0
    x = np.fft.irfft(spec, n)
    x /= np.max(np.abs(x)) * 1.05
    return x.astype(np.float32)


def verdict_of(x, sr=SR):
    return analyze_transcode_evidence(x, sr).evidence


def test_full_band_passes():
    e = verdict_of(noise_track())
    assert e.verdict == "PASS", f"clean noise scored {e.lossy_score:.0f}"
    assert e.shelf_type == "none"


def test_brickwall_fails():
    for cutoff in (16000, 18600, 20000):
        e = verdict_of(noise_track(cutoff=cutoff))
        assert e.verdict == "FAIL", f"{cutoff} Hz wall only scored {e.lossy_score:.0f}"
        assert abs(e.cutoff_freq - cutoff) < 600, f"located the wall at {e.cutoff_freq:.0f}"
        assert e.shelf_type == "hard"


def test_silence_is_inconclusive():
    e = verdict_of(np.zeros(SR * 5, dtype=np.float32))
    assert e.verdict == "INCONCLUSIVE"


def test_stereo_matches_mono():
    mono = noise_track(cutoff=16000)
    stereo = np.stack([mono, mono], axis=1)
    assert verdict_of(stereo).verdict == verdict_of(mono).verdict


def test_scores_are_monotonic():
    slopes = [-5, -12, -20, -35, -60, -120]
    scored = [_score_slope(s) for s in slopes]
    assert scored == sorted(scored)
    assert 0.0 <= scored[0] and scored[-1] <= 100.0
    assert _score_slope(90) == 0.0

    depths = [0, 5, 8, 12, 18, 40]
    scored = [_score_shelf_depth(d) for d in depths]
    assert scored == sorted(scored)
    assert scored[0] == 0.0 and scored[-1] == 100.0


def test_shelf_naming():
    assert shelf_type_from_slope(-90, 20) == "hard"
    assert shelf_type_from_slope(-40, 12) == "medium"
    assert shelf_type_from_slope(-25, 12) == "soft"
    # a gentle taper is not a shelf however you squint at it
    assert shelf_type_from_slope(-25, 3) == "none"
    assert shelf_type_from_slope(-15, 12) == "soft"
    assert shelf_type_from_slope(-2, 20) == "none"


def test_shortest_inputs():
    assert verdict_of(np.zeros(1, dtype=np.float32)).verdict == "INCONCLUSIVE"
    try:
        verdict_of(np.zeros(0, dtype=np.float32))
    except ValueError as ex:
        assert str(ex) == "audio has no samples"
    else:
        assert False, "empty audio should fail clearly"


def test_compare_alignment():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(200)
    shift = 7

    b_late = np.r_[np.zeros(shift), a[:-shift]]
    aa, bb, lag = _align_signals(a, b_late, sample_rate=100)
    n = min(len(aa), len(bb))
    assert lag == -shift and np.allclose(aa[:n], bb[:n])

    b_early = np.r_[a[shift:], np.zeros(shift)]
    aa, bb, lag = _align_signals(a, b_early, sample_rate=100)
    n = min(len(aa), len(bb))
    assert lag == shift and np.allclose(aa[:n], bb[:n])

    silent = np.zeros(200)
    aa, bb, lag = _align_signals(silent, silent, sample_rate=100)
    assert lag == 0 and len(aa) == len(bb) == len(silent)


def test_jitter_reads_the_ceiling_not_the_content():
    # most frames sit low, a handful reach a hard ceiling at 16k
    edges = np.concatenate([np.linspace(8000, 12000, 90), np.full(30, 16000.0)])
    active = np.ones(len(edges), dtype=bool)
    assert compute_edge_jitter(edges, active) < 50


def test_clip_runs_groups_and_caps():
    x = np.zeros(SR)
    x[100:400] = 1.0        # one long run
    x[SR // 2:SR // 2 + 5] = 1.0
    times = clip_runs(x, SR)
    assert len(times) == 2
    assert abs(times[0] - 100 / SR) < 1e-6


def test_fmt_time():
    assert fmt_time(0.05) == "50ms"
    assert fmt_time(2.5) == "2.50s"
    assert fmt_time(75).startswith("1m")


def test_retained_display_frames_match_full_stft():
    sr = 48000
    seconds = 10
    frequency = 6000
    t = np.arange(sr * seconds) / sr
    mono = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    stereo = np.column_stack([mono, mono])

    frames = analyze_display_array(stereo, sr, max_time_bins=80)
    spec = compute_display_spectrogram(frames)

    mono = stereo.mean(axis=1)
    freqs_old, times_old, Zxx = stft(
        mono, fs=sr, nperseg=1024, noverlap=512, window='hann')
    decimation = max(1, Zxx.shape[1] // 80)
    old_db = 10 * np.log10(np.abs(Zxx[:, ::decimation]) ** 2 + 1e-10)

    assert frames.frame_count == len(stereo)
    assert frames.full_stft_frames == Zxx.shape[1]
    assert frames.time_decimation == decimation
    assert np.array_equal(spec.frequencies, freqs_old)
    assert np.array_equal(spec.times, times_old[::decimation])
    assert np.allclose(spec.Sxx_db, old_db, rtol=0, atol=2e-4)
    preview_spec = compute_display_spectrogram(frames, preview_floor=True)
    old_preview_db = 20 * np.log10(np.abs(Zxx[:, ::decimation]) + 1e-10)
    visible = old_preview_db >= np.max(old_preview_db) - 80
    assert np.allclose(preview_spec.Sxx_db[visible], old_preview_db[visible],
                       rtol=0, atol=2e-4)
    old_pool = _pool(old_preview_db, 36, 80)
    new_pool = _pool(preview_spec.Sxx_db, 36, 80)
    old_norm = np.clip((old_pool - (old_pool.max() - 80)) / 80, 0, 1)
    new_norm = np.clip((new_pool - (new_pool.max() - 80)) / 80, 0, 1)
    assert np.array_equal((new_norm * 15).astype(int),
                          (old_norm * 15).astype(int))
    peak_hz = spec.frequencies[np.argmax(np.mean(spec.Sxx_db, axis=1))]
    assert abs(peak_hz - frequency) <= spec.frequency_resolution

    shortest = analyze_display_array(np.ones(1, dtype=np.float32), sr)
    shortest_spec = compute_display_spectrogram(shortest)
    assert shortest_spec.Sxx_db.shape == (1, 1)
    assert np.all(np.isfinite(shortest_spec.Sxx_db))


def test_streaming_dynamics_match_in_memory():
    rng = np.random.default_rng(14)
    data = (rng.standard_normal((123457, 2)) * 0.2).astype(np.float32)
    data[100:400, 0] = 1.0
    data[50000:50010, 1] = -0.995
    expected = analyze_dynamics(data, SR)

    accumulator = DynamicsAccumulator(SR, 2)
    for start in range(0, len(data), 7777):
        accumulator.update(data[start:start + 7777])
    actual = accumulator.finish()

    assert abs(actual.peak_db - expected.peak_db) < 1e-5
    assert abs(actual.rms_db - expected.rms_db) < 1e-5
    assert abs(actual.dynamic_range - expected.dynamic_range) < 1e-5
    assert actual.clipped_samples == expected.clipped_samples
    assert actual.clip_times == expected.clip_times


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as ex:
            fails += 1
            print(f"FAIL  {name}: {ex}")
    sys.exit(1 if fails else 0)
