# Changelog

# v2.0.0

- Accuracy: slope detection rewritten as steepest 250 Hz drop instead of a symmetric regression window, which was diluting brick walls with passband (a -172 dB/kHz mp3 shelf measured as -10.8).
- Accuracy: cutoff candidates now every 250 Hz from 13-21.5k instead of seven hand-picked spots, hard-cutoff drop threshold 15 -> 10 dB so shelves in already-quiet top end still register.
- Accuracy: codec matching now uses the p97 active edge (codec ceiling) instead of p90, which tracks content and sits way under the lowpass on quiet material.
- Accuracy: classic-signature rule, brick wall at a codec anchor that persists = minimum 75 score. A real mp3-128 transcode hidden in FLAC now FAILs (was PASS 27).
- Accuracy: slope search capped at 21 kHz so natural content edges and anti-alias filters on hi-res files stop inflating clean scores.
- Honest limitation, validated against real transcodes: high-bitrate codecs (mp3-320, opus-128, AAC ~256) overlap with dark genuine masters on these features and still PASS. The README warning is real.
- Performance: detect analyzes the middle 150s instead of the whole file (codec lowpass is global), 50% STFT overlap, detect plot at 110 dpi. 96/24 6-minute file: 15s -> 6s. Normal album track: ~2.5s.
- Detect plot redesigned: dark theme, monospace, verdict-colored header with scores, accent palette in plotting.py. Basic and compare plots inherit it.
- Verdict and warning wording rewritten to sound like a person.
- Profile resemblance section (table and batch column) hidden when there is no cutoff to compare against, ranking codecs on a full-band file was noise.
- Fixed: effective_bit_depth crashed on full-scale samples (INT_MIN lowest-bit trick went negative).
- New JSON field: active_edge_p97_hz.
- Added -p / --preview: renders the spectrogram straight in the terminal (256-color, ASCII fallback). No matplotlib, no PNG, done in milliseconds.
- Added batch detect: `spectro *.flac --detect` analyzes every file and prints a color-coded verdict table; --json writes a combined report.
- Added effective bit-depth check: catches 16-bit masters padded into 24/32-bit containers by inspecting the lowest set bit.
- Added fake-stereo check: flags channels with ~1.0 correlation (mono upmix).
- Added -v / --version with ASCII spectrogram banner plus python/numpy/scipy/soundfile/matplotlib versions.
- Detect output now color-coded: verdicts, severity tags, warnings, and score bars for lossy/quality/subscores. Respects NO_COLOR and non-tty pipes.
- Running spectro with no arguments now prints a usage summary before prompting; Ctrl-C / empty input exit cleanly instead of erroring.
- --verbose no longer owns -v (now long-form only).
- Fixed: jitter no longer counts as strong evidence on its own, a dead-stable natural rolloff (analog masters) was triggering a HIGH "locked edge" flag on clean files. Now capped unless edge or slope show something codec-like.
- Fixed: "Closest cutoff resemblance" now says none on PASS files without a hard cutoff instead of naming a random codec.
- Fixed: bare --json wrote nothing because the const collided with the default; default report name works again.
- Fixed: slow-import hint never fired (numpy was already imported before the timer started).
- Fixed: edge percentiles (p10/p50/p90) now computed over active frames only, so silence no longer drags the spectral edge down and skews edge/quality scoring.
- Fixed: all-silent files no longer hit a NaN warning path in edge percentile computation.
- Fixed: missing-file error no longer claims the file "is not an audio file".
- Performance: vectorized rolloff-85 variance (dropped per-frame Python loop).
- New JSON fields: effective_bit_depth, stereo_correlation.
- Docs: MANUAL updated for --preview, --version, batch mode, and no-arg behaviour.

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
- Added --json FILE for json reports.
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