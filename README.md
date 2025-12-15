# spectro

Small Python tool to generate spectrograms + (optional) do a heuristic “lossy source” check.

It makes pictures. It does **not** prove provenance, codec, or bitrate.

## Install

Python 3.9+ recommended.

```bash
pip install numpy soundfile scipy matplotlib
```

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

## What the modes actually do

- default (fast): caps sample rate at 44.1k when needed, smaller FFT, lower DPI, saves PDF by default
- `--quality`: no downsampling, bigger FFT + overlap, higher DPI, saves PNG by default
- `--detect`: estimates cutoff + shelf behavior and prints the closest matching profile (heuristic), then saves a 1-page analysis PDF
- `--compare`: time-aligns two files, computes similarity %, and renders A, B, and difference spectrograms

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

That’s it.