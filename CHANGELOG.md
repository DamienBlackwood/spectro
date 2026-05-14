# Changelog

# v1.5.0

- Detect mode rewritten around a six-feature evidence-score system instead of hard cutoffs and fallback-heavy heuristics.
- Verdicts now use independent lossy_score and quality_score metrics.
- Silent, narrowband, and analog sources now land in INCONCLUSIVE instead of being force-classified.
- Added spectral-edge, slope, jitter, band-ratio, rolloff-stability, SBR, and persistence scoring.
- Added built-in MP3 and Opus cutoff reference anchors.
- Verdict thresholds rebalanced for PASS / WARN / FAIL / INCONCLUSIVE.
- Added --verbose / -v for per-feature subscore output.
- Refactor: split the old 1100-line spectrogram.py into the spectro/ package.
- Refactor: all heuristic constants now live in a frozen Thresholds dataclass.
- Refactor: module imports no longer trigger side effects or CLI execution.
- Added JSON fields for scores, subscores, jitter, rolloff variance, slope, and band ratio.
- Removed unused transcode_suspected.
- Removed leftover confidence / confidence_score fields.
- Performance: reused STFT frames for window analysis instead of recomputing FFTs.
- Performance: vectorized active_band_edge with NumPy.
- Performance: Matplotlib now lazy-loads only when plotting.
- Fixed: FAIL plot titles no longer match WARN titles.
- Fixed: duplicate argparse parsers consolidated into build_parser().
- Fixed: duplicate audio validation merged into looks_like_audio(path).
- Fixed: JSON flags are now structured {severity, name, detail} objects.
- Fixed: _analyze_channel now returns a ChannelAnalysis dataclass instead of an 8-tuple.
- Fixed: spectrogram + edge analysis now average channels consistently.
- Fixed: classify_transcode no longer returns (evidence, flags) tuples.

# v1.4.5

- Refactor: project converted into the spectro CLI application.
- Results now save beside the input file instead of a dedicated output directory.
- Output switched to PNG-only for faster saves and smaller files.
- Added automatic result opening unless --no-open is passed.
- Added FFmpeg fallback loading for unsupported formats like MP3.
- Migrated packaging to pyproject.toml with setuptools entry points.
- Verdict system replaced with PASS / WARN / FAIL / INCONCLUSIVE.
- Renamed detect_codec() to analyze_transcode_evidence().
- Renamed CodecResult to SpectralAnalysisResult.
- Added per-channel analysis instead of mono downmixing.
- Detection STFT increased to 8192 bins.
- Added active-window filtering for persistence and edge metrics.
- Added ~5-second time-local cutoff scanning.
- Added active spectral-edge percentile reporting.
- Added cutoff persistence tracking tied to candidate frequencies.
- Added band-drop scoring across 14–21 kHz candidate bands.
- Replaced hardcoded cutoffs with Nyquist-relative thresholds.
- SBR detection now reports none, possible, or likely.
- High sample-rate files without ultrasonic content now warn instead of auto-failing.
- Console output now prints a limitations section after analysis.
- Added --json FILE for machine-readable reports.
- Added EvidenceFlag severity levels.
- Added a three-panel forensic analysis plot.
- Added bit-depth reporting from ffprobe.
- Removed --quality.
- Removed --no-display.
- Removed --open.
- Removed lossless reference profiles from CODEC_PROFILES.
- Removed codec-confidence scoring in favor of evidence heuristics.
- Removed is_lossless from the result dataclass.
- Performance: compare mode now derives difference STFT directly from Z1/Z2.
- Fixed: clip_times now returns Python lists.
- Fixed: FFmpeg availability is checked before fallback execution.
- Fixed: compare-mode similarity now uses correlation coefficient instead of RMS.
- Fixed: lower cutoffs now sort as more suspicious.
- Fixed: active edge floor margin raised to 18 dB.
- Fixed: near-Nyquist and high-band ranges now use corrected bounds.

# v1.4.0

- Spectrum plots now extend to full Nyquist frequency instead of stopping at 22 kHz.
- Added --open for automatic result opening.
- Added dynamic range analysis.
- Added clip detection.
- Added loudness-war-era classification.

# v1.3.0

- Removed the 24 kHz downsampling limit in detect mode.
- Added ultrasonic-content analysis for files above 48 kHz.
- Added metadata extraction through ffprobe.
- Performance: decimation now happens immediately after STFT generation to reduce memory usage.

# v1.2.0

- Added --compare for side-by-side analysis.
- Codec detection completely rewritten around gradient analysis.
- Added lossless, transcode, and upsample detection.
- Reduced redundant STFT calls in detect mode.

# v1.1.0

- Switched to a vectorized STFT implementation.
- Simplified commands.
- Replaced home-grown DSP algorithms with proper libraries.

# v1.0.0

Initial release.

- Spectrogram generation for audio analysis.
- Codec and transcode inspection tooling.
- FFT/STFT-based spectral analysis pipeline.
- Command-line forensic audio inspection.