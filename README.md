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

Common flags:

```bash
-o out.pdf        # custom output name
--log             # log frequency axis (visual aid)
--no-display      # don’t open the window
--info            # print file info then exit
```

## What the modes actually do

- default (fast): caps sample rate at 44.1k when needed, smaller FFT, lower DPI, saves PDF by default
- `--quality`: no downsampling, bigger FFT + overlap, higher DPI, saves PNG by default
- `--detect`: estimates cutoff + shelf behavior and prints the closest matching profile (heuristic), then saves a 1-page analysis PDF

## Output

Files go to:

```
outputs/spectrograms/
```

## Understanding “detect” results (please read!)

- “closest profile” means “looks most similar to”, not **“is”**
- a clean horizontal cutoff is a strong hint of lossy processing
- no cutoff detected ≠ guaranteed lossless

That’s it.