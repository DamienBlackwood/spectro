# spectro

Small Python tool to generate spectrograms + (optional) do a heuristic “lossy source” check.

It makes pictures. It does **not** prove provenance, original sample rate, codec history, or bitrate.

## Install

**Requirements:** Python 3.9+

Clone the repo, set up a virtual environment, and install dependencies:

```bash
git clone <repo-url>
cd spectro
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -U pip
pip install numpy soundfile scipy matplotlib
```

### Optional: FFmpeg / ffprobe

Helps cross-check codec, sample rate, and bit depth from file metadata.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: download static build and add bin/ to PATH
```

(FFmpeg installs system-wide, not in the venv.)

## Usage

Fast spectrogram (default):

```bash
python spectrogram.py "song.flac"
```

High quality spectrogram (slower):

```bash
python spectrogram.py "song.flac" --quality
```

Heuristic lossy-source analysis + report PDF:

```bash
python spectrogram.py "song.flac" --detect
```

Compare two files (waveform + spectrogram difference):

```bash
python spectrogram.py "A.flac" --compare "B.flac"
```

Show file info only (no plots):

```bash
python spectrogram.py "song.flac" --info
```

Common flags:

```bash
-o out.pdf        # custom output name
--detect          # heuristic lossy-source analysis + 1-page PDF
--compare FILE    # compare against another file (spectrogram + diff)
--quality         # full-quality spectrogram (slower)
--log             # log frequency axis (visual aid)
--no-display      # don’t open the window
--info            # print file info then exit
```

## How the modes work

- **default (fast)**: smaller FFT, lower DPI, PDF output
- **--quality**: full-resolution FFT + overlap, higher DPI, PNG output
- **--detect**: analyzes frequency cutoffs and shelf behavior; outputs a 1-page analysis PDF
- **--compare**: time-aligns two files, shows similarity %, renders both spectrograms + difference

## Output

Files go to:

```
outputs/spectrograms/
```

Naming defaults:
- fast / quality: `<input>.(pdf|png)`
- `--detect`: `<input>_analysis.pdf`
- `--compare`: `<input>_vs_<other>.pdf`

## Understanding “detect” results (please read!)

- “closest profile” means “looks most similar to”, not **“is”**
- a clean horizontal cutoff is a strong hint of lossy processing
- no cutoff detected ≠ guaranteed lossless
- all results are probabilistic; false positives and negatives are expected
- “likely lossless” means “no obvious lossy low-pass or shelf detected”
- bandwidth-limited material (e.g. ~20–22 kHz) inside a higher sample-rate container is common
- absence of ultrasonic energy does **not** imply fakery or lossy transcoding
- classic and archival recordings often show this pattern even when transfers are legitimate

That’s it.