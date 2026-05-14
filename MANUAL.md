# SYNOPSIS

`spectro` [*FILE*] [*OPTION* ...]

`python spectrogram.py` [*FILE*] [*OPTION* ...]

# OPTIONS

`--detect`
:   Run spectral forensics. Computes a 0–100 *lossy evidence score* from six independent signals (active spectral edge percentiles, transition band slope, edge jitter over time, high/low band energy ratio, rolloff stability, SBR likelihood). Along with a separate 0–100 *data quality score* that flags silent, narrow band, or analog source material.

 Verdict (PASS / WARN / FAIL / INCONCLUSIVE) is derived from both.


`--compare` *FILE*
:   Compare two audio files side-by-side with a null-test difference spectrogram.

`--log`
:   Use logarithmic frequency scale.

`-o`, `--output` *FILE*
:   Save spectrogram as specified filename.

`--no-open`
:   Do not open automatically.

`--info`
:   Only display file info.

`--json` [*FILE*]
:   Write a JSON report alongside the analysis plot. If no filename given, defaults to `[input_stem]_analysis.json`. The report includes `lossy_score`, `quality_score`, per-feature `subscores`, and all underlying metrics.

`-v`, `--verbose`
:   In `--detect` mode, print every subscore to see what gave the final response.

# EXAMPLES

`spectro track.flac`
:   Simple spectrogram generation.

`spectro track.flac --detect`
:   Runs *"forensics"* and prints verdict + lossy/quality scores.

`spectro track.flac --detect --verbose`
:   Same as above, but with per feature subscore breakdown (edge, slope, jitter, high_band, rolloff, sbr, persistence).

`spectro orig.flac --compare remaster.flac`
:   Compares two audio files.

`spectro track.wav --log -o analysis.png`
:   Logarithmic frequency scale with custom filename.

# FILES

All output files are written to the same directory as the input audio file.

`[input_directory]/[filename].png`
:   Default spectrogram output.

`[input_directory]/[filename]_analysis.png`
:   Plot produced by `--detect` mode (3-panel: spectrum, spectrogram, active edge).

`[input_directory]/[filename]_vs_[other].png`
:   Comparison plot produced by `--compare`.

`[input_directory]/[filename].json` (or custom name)
:   Machine-readable report when `--json` is passed in `--detect` mode.

# INSTALLATION

Requires Python 3.8+. FFmpeg is recommended for MP3 and other formats soundfile cannot read.

**Recommended (pipx):**

```bash
git clone https://github.com/DamienBlackwood/spectro.git
cd spectro
pipx install .
```

Then run `spectro` from anywhere. Note that the first launch after install may take a few seconds — numpy, scipy, and matplotlib need to initalise Subsequent launches are snappy!

**Manual:**

```bash
pip install numpy matplotlib scipy soundfile
python spectrogram.py track.flac
```
