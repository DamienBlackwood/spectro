# Changelog

# v2.0.0

Everything below was measured against a corpus of 14 known-lossless tracks spanning 1965 to 2023 and 44.1/48/96/192 kHz, put through eight encoders and decoded back to FLAC. 126 files.

Before: 12/14 clean (two genuine files were FAILing), 51/112 transcodes flagged. After: all 14 clean references are unflagged (13 PASS, one INCONCLUSIVE) and 54/112 transcodes are WARN/FAIL. Eight more transcodes are INCONCLUSIVE because their source is the same narrow-band orchestral track; they are not counted as catches. Clean files used to score up to 75 out of 100 against a WARN line of 45; they now top out at 43.

What still slips through is mp3 V0 and Apple aac 256, which genuinely have no cutoff left to find. Full table in the README.

- Performance, with no analysis or rendering approximation: PNG and terminal modes scan every sample once for exact dynamics, then compute the exact STFT frames that their original post-STFT decimation retained. Same Hann window, centres, zero-padding, scaling, timestamps, frequency bins, and dB values; discarded FFT frames are never created. On a 525 MiB, 192 kHz, 13m33s stereo FLAC, PNG dropped from 2m09s to 4.2s and terminal preview from 1m24s to 3.8s. The optimized PNG is pixel-identical to the original 1800x900 Matplotlib output.
- Performance: exact dynamics now use one bounded-memory streaming pass instead of seven whole-array passes and multi-gigabyte temporaries. The same stress file's dynamics pass dropped from 11.7s after loading to 3.5s including FLAC decoding.
- Performance: Spectro seeds an isolated 1.7 KiB Matplotlib cache with the same bundled DejaVu faces used by ASCII-titled plots, removing the first-render system-font crawl with pixel-identical output. Non-ASCII titles retain full system-font discovery and fallback.
- Accuracy: the slope feature was measuring spectral raggedness, not shelves. Its smoothing kernel was a bin count, so it was 16 Hz wide at 44.1 kHz and 70 Hz at 192 kHz and the same master measured differently depending on its container. Under-smoothed, one ragged notch in a quiet top end reads as a brick wall, which is how two genuine 44.1 kHz files were coming back FAIL at -62 and -35 dB/kHz. Kernel is a fixed 60 Hz now and the slope thresholds are recalibrated to match.
- Accuracy: the "brick wall at a codec anchor" shortcut to a 75 score now also requires the shelf to have real depth. A steep wall with nothing behind it is a resampler's anti-alias filter, which every 44.1 kHz downsample has.
- Accuracy: files whose content stops below every codec anchor now come back INCONCLUSIVE instead of PASS. If a track runs out at 9 kHz a lowpass could be sitting anywhere above it and leave no trace, so PASS was claiming more than it knew.
- Accuracy: added a 17.5 kHz codec anchor, where both ffmpeg's and Apple's AAC put their 128 kbps cutoff. No clean file in the corpus sits in that window.
- Accuracy: the reported cutoff was wrong on most files. The gradient walk latched onto the first dip past 12 kHz, so a clean 48 kHz master and an mp3-320 both came back "15009 Hz". Cutoff and shelf type now come from the deepest measured shelf and the measured slope, which also fixes the "Closest cutoff resemblance" line (it named vorbis_128 for nearly everything) and un-breaks SBR detection, which was gated on that same bad number.
- Accuracy: new *shelf depth* subscore, how far the spectrum falls across the cutoff. Clean files measure under 6 dB, transcodes 9-24. It took over the weight that rolloff variance was wasting.
- Accuracy: rolloff-85 variance dropped from the score. It measured ~1100-1250 Hz on clean and transcoded copies of the same track, so it was 10% of the budget contributing nothing. Still reported, just not counted.
- Accuracy: edge jitter is now read off the top quartile of active frames instead of all of them. On dynamic material the old figure measured how much the content moved (1500 Hz on a brickwalled opus file) rather than how tightly the codec ceiling held.
- Accuracy: a shelf under 10 dB deep now still counts as a hard cutoff if the slope beside it is steeper than -32 dB/kHz. Analog masters have so little top end that a brick wall only costs 6-9 dB, and opus-128 was passing because of it.
- Accuracy: jitter only scores when the edge sits near a codec anchor *and* there is a shelf under it. A dead-stable natural rolloff is not evidence of anything.
- Accuracy: weights rebalanced toward slope and shelf depth, the two features that actually separate. edge 25→20, slope 20→25, shelf 0→20, jitter 15→10, high_band 15→10.
- Accuracy: codec cutoff ranges in CODEC_PROFILES re-measured from real encodes instead of estimated. Added mp3_v0, aac_192, aac_256. Profile ties now break on whichever centre is closest to the measured cutoff.
- Added `tests/make_corpus.py`: point it at your own lossless files and it builds the transcode set, scores everything and tells you where it was wrong. This was the "build a test corpus" TODO.
- Added `tests/test_detect.py`: synthetic signals, no audio needed, catches anyone inverting the scoring in a refactor.
- Fixed: `--preview` grid alignment on the default macOS terminal. Block characters are East Asian "ambiguous" width and Terminal.app draws them double-wide; it now falls back to a coloured ASCII ramp there, overridable with `SPECTRO_BLOCKS`.
- Added half-block rendering to `--preview`, double the vertical resolution, and a framed axis. Colour runs are coalesced so the output is a third smaller.
- Fixed: clip detection ran on a mono downmix, so out-of-phase clipping cancelled and never showed up. It also built a Python list of every clipped sample before throwing all but 8 away. Now per-channel, and it reports where each clipped *run* starts rather than eight consecutive samples of the same one.
- Fixed: bit depth printed as "0-bit" for lossy files, because ffprobe returns the string "0" and that is truthy.
- Fixed: `--compare` used FFT resampling on whole songs, now polyphase. Also prints a per-band level difference (low/mid/high/air) so the comparison says something numeric.
- Fixed: matplotlib figures were never closed in detect and compare.
- Added `--fails-only`, `--sort` and `-j/--jobs` for batch detect. Batch now runs 4 files at once by default, about 1.5x on an album. Threads, not processes: a process pool measured slower because each worker pays a fresh numpy import.
- Every mode accepts multiple files now, not just `--detect`. They run one after another.
- Ctrl-C exits cleanly instead of dumping a traceback. Piping into `head` no longer errors either.
- Conflicting flags now say which one won instead of silently ignoring the rest, and `--json` says so when it isn't going to write anything.
- Interactive prompt now handles paths dragged into a terminal, where spaces come through backslash-escaped.
- `-v` lists dependency versions from a tuple instead of by hand, and says whether ffmpeg is around.
- `--info` shows sample rate, channels and duration.
- `import spectro` no longer pulls in numpy, scipy and matplotlib just to read `__version__`.
- Audio sniffing recognises m4a/mp4 `ftyp` boxes, raw mp3/aac frame headers, and a few more extensions. Dropped MIDI, which is not audio soundfile can read.
- Removed dead code: `compute_slope_db_per_khz`, `main_wrapper`, `_build_evidence_shell` and six unused thresholds left over from the pre-v1.5 verdict system.
- New JSON fields: `cutoff_hz`, `shelf_type`.

(below this is 2.0.0 i forgot to commit it)
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
