# Changelog

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