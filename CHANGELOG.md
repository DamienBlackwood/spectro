# Changelog

## v1.5.0

Rewrote detect mode with a much smarter verdict system, plus a big internal cleanup. Old `--detect` was conservative to a fault, anything that wasn't an obvious hard cutoff fell into INCONCLUSIVE. Replaced it with a six-feature evidence-score system based on actual codec literature.

- Verdict now uses two scores: `lossy_score` (0–100) and `quality_score` (0–100), computed independently. Silent / narrowband / analog sources land in INCONCLUSIVE instead of getting force-classified into PASS or FAIL.
- Six features feed into `lossy_score`: spectral edge (p90 distance from known codec cutoffs), transition-band slope in dB/kHz, edge jitter over time (MAD), high/low band energy ratio, rolloff stability, SBR likelihood, and cutoff persistence.
- Codec cutoff anchors built in: 16k / 18.6k / 19.7k / 20.5k for MP3 128/192/256/320, plus Opus bandwidth modes.
- Balanced verdict thresholds: FAIL ≥ 70 with quality ≥ 60, WARN ≥ 45 with quality ≥ 50, INCONCLUSIVE when quality < 50, PASS otherwise.
- Flags now come from the subscores with severity tagging, so you can see which signal triggered the verdict.
- Added `--verbose` / `-v` to print every per-feature subscore.
- Split the 1100-line `spectrogram.py` into a flat `spectro/` package (`cli.py`, `audio.py`, `dynamics.py`, `spectral.py`, `verdict.py`, `plotting.py`, `reporting.py`, `dataclasses_.py`, plus a `commands/` subpackage). Root `spectrogram.py` is now a six-line wrapper for backwards compatibility.
- All heuristic magic numbers (~50 of them) now live in a frozen `Thresholds` dataclass.
- Module is importable without side effects now, so `import spectro.spectral` works in notebooks and tests.

### Added

- `--verbose` / `-v` flag for per-feature subscore breakdown in `--detect` mode.
- `lossy_score`, `quality_score`, `subscores`, `edge_jitter_hz`, `rolloff_85_var_hz`, `max_slope_db_per_khz`, `band_ratio_db` in the JSON report.

### Removed

- `transcode_suspected` field (was set but never read).
- `confidence` / `confidence_score` fields. v1.4.5 changelog claimed these were gone, they weren't. They are now.
- Module-level argparse + `sys.exit` block at top of `spectrogram.py`.

### Performance improvements

- `analyze_time_windows` slices the existing STFT frames along the time axis instead of recomputing a fresh STFT per 5-second window. Saves ~60 redundant FFTs on a 5-minute track.
- `active_band_edge` vectorized, replaced the per-frame Python loop with NumPy ops.
- Matplotlib import is lazy now, only loads when actually plotting.

### Fixes

- FAIL plot title was identical to WARN, fixed.
- Duplicate argparse parser (one at module top, one inside `main()`) consolidated into a single `build_parser()`.
- Duplicate audio-magic check consolidated into a single `looks_like_audio(path)` helper.
- JSON flags are now structured `{severity, name, detail}` objects instead of pre-formatted strings.
- `_analyze_channel` returns a `ChannelAnalysis` dataclass instead of an 8-tuple.
- `Sxx_db` averaged across channels before driving the spectrogram + edge analysis (previously the spectrogram panel and spectrum line could disagree on mid/side-mastered stereo).
- `classify_transcode` no longer returns a `(evidence, flags)` tuple, flags are attached to the evidence directly.


## v1.4.5

Refactor + forensic evidence pipeline. No more codec identity claims just transcode/upsampling evidence.

- Refactored to CLI application, installed with `pipx install .`. The name is now `spectro`.
- Results saved alongside the input, rather than in a specific folder.
- PNG-only output; faster saves and smaller files.
- Opens the resulting file automatically upon successful completion, unless the `--no-open` flag is passed.
- Implemented `load_audio()` with fallback to FFmpeg when working with MP3 and other unsupported formats.
- Uses `pyproject.toml` with setuptools entry point.
- Verdict system updated to PASS / WARN / FAIL / INCONCLUSIVE.
- `detect_codec()` renamed to `analyze_transcode_evidence()`; `CodecResult` renamed to `SpectralAnalysisResult`.
- Per-channel detection instead of mono downmixing.
- Detection STFT bumped to 8192 bins.
- Active window filtering: persistence and edge metrics computed only from loud, spectrally rich frames.
- Time-local detection: ~5-second windows scanned for suspicious cutoff frequencies.
- Active band edge per frame, reported as p10/p50/p90 percentiles.
- Cutoff persistence tied to the suspected cutoff frequency rather than a generic Nyquist threshold.
- Band-drop scoring across candidate frequencies (14–21 kHz).
- Nyquist-relative cutoff frequency thresholds (0.92×, 0.94×) instead of hardcoded 22 kHz.
- SBR detection nuanced to `none` / `possible` / `likely` using correlation + spectral flatness.
- High sample-rate files with no ultrasonic activity receive a warning flag instead of a lossy label.
- Console output prints a limitations block after every analysis.

### Added

- `--json FILE` flag for machine-readable JSON reports.
- EvidenceFlag dataclass with severity levels (info / low / medium / high / critical).
- 3-panel forensic plot: average spectrum, spectrogram, and active spectral edge over time.
- Bit depth shown in `--detect` output when available from ffprobe.

### Removed

- `--quality` parameter. Since both modes use identical hop sizes, the time resolution is actually the same.
- `--no-display` parameter. Couldn't work properly, as Agg backend can't handle `plt.show()`.
- `--open` parameter.
- `lossless` profile from `CODEC_PROFILES`; only lossy reference profiles are left for resemblance scoring.
- Confidence score for codec identity; replaced with evidence-based flags and metrics.
- `is_lossless` field from result dataclass.

### Performance improvements

- SciPy imports moved to module scope.
- Compare mode derives the difference STFT from Z1 and Z2 directly instead of computing a third STFT.
- Cleaned `del` operations and removed unused variables.

### Fixes

- `clip_times` now returns a list, not a numpy array.
- `load_audio()` makes sure ffmpeg binary exists before executing.
- Similarity calculation in compare mode is changed from RMS to correlation coefficient.
- `suspicion_key` now sorts lower cutoff as more suspicious (was inverted).
- `active_band_edge` floor margin raised to 18 dB.
- `near_nyquist_db` band fixed to `nyquist * 0.90`–`0.99`; `high_band_db` capped at `min(20000, nyquist * 0.90)`


## v1.4.1

- Improved `README.md`

## v1.4.0

- Show the full spectrum range till the Nyquist, rather than being limited to 22kHz in normal mode.
- Added `--open` flag to automatically open the result file.
- Dynamic range analysis added. Also includes clip detection and loudness war era classification.

## v1.3.0

- Removal of downsampling limitation to 24kHz in detect mode.
- Check whether the file contains ultrasonic content (CD upsampled, for example) if sampling frequency above 48kHz.
- Extraction of metadata (ffprobe is required).
- Memory fixes: apply decimation immediately after STFT to avoid building huge arrays.

## v1.2.0

- `--compare` flag added for side-by-side comparison.
- Complete rewrite of codec detection with gradient analysis.
- Lossless file detection and detection of transcoded/upsampled lossy files added.
- Fewer STFT calls in detect mode.

## v1.1.0

- Vectorized STFT implementation used.
- Commands simplified.
- Replaced home-brew algorithms with proper libraries.

## v1.0.0

- Initial release.