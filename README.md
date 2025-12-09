# Spectrogram Generator

A high-performance, multi-threaded audio spectrogram generation tool with intelligent optimization and minimal, elegant terminal UI. Convert audio files to beautiful spectrograms with smart quality settings and performance tuning.

## Features

- **Multi-threaded STFT computation** - Parallel processing with automatic thread optimization for large files
- **Multiple output formats** - PNG, PDF, SVG with automatic format switching for speed
- **Intelligent time estimation** - Learning-based predictions that improve accuracy over time
- **Professional file organization** - Automatic folder structure with metadata-based naming
- **Comprehensive audio support** - 20+ lossless formats including FLAC, WAV, AIFF, APE, WavPack
- **Emergency speed mode** - 50x faster generation by bypassing matplotlib rendering
- **Smart performance optimization** - Auto-detects optimal settings based on file size and duration
- **Metadata extraction** - Automatic artist-title filenames from audio tags
- **Minimal aesthetic UI** - Clean progress bars with smooth animations (via alive-progress)
- **Detailed audio analysis** - Complete file information including bitrate, sample rate, bit depth
- **Timeout protection** - 30-second save timeout prevents infinite hangs
- **Annotation support** - Add custom markers and labels to spectrograms via JSON

## Quick Start

### Installation

1. **Clone and setup virtual environment:**
```bash
cd spectro
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or on Windows:
# venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install soundfile numpy matplotlib pillow alive-progress mutagen
```

3. **Verify installation:**
```bash
python spectrogram.py --hotkeys
```

### Basic Usage

```bash
# Generate spectrogram from audio file
python spectrogram.py song.mp3

# Custom output filename
python spectrogram.py "My Song.flac" -o my_spectrogram.png

# Show all available options
python spectrogram.py --hotkeys

# Ultra-fast generation
python spectrogram.py audio.wav --emergency

# High-quality with logarithmic scale
python spectrogram.py song.flac --quality high --log
```

## Installation & Requirements

### System Requirements
- Python 3.8+ (tested with 3.13.5)
- macOS, Linux, or Windows with command line access

### Core Dependencies
| Package | Purpose | Version |
|---------|---------|---------|
| `soundfile` | Audio file reading | Latest |
| `numpy` | Numerical computations (STFT) | Latest |
| `matplotlib` | Visualization and rendering | Latest |
| `pillow` (PIL) | Image processing and direct rendering | Latest |
| `alive-progress` | Beautiful progress bars | 3.3.0+ |
| `mutagen` | Audio metadata extraction | Latest (optional) |

### Optional Dependencies
- `mutagen` - For better metadata extraction (artist, title, album)
- `tinytag` - Fallback metadata extraction if mutagen unavailable

### Complete Installation Steps

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux

# Install all requirements
pip install soundfile numpy matplotlib pillow alive-progress mutagen
```

## Usage Guide

### Basic Commands

```bash
# Simple spectrogram generation
python spectrogram.py audio.mp3

# Specify output filename and format
python spectrogram.py audio.mp3 -o custom_name.pdf

# Show file information before processing
python spectrogram.py audio.wav --info
```

### Speed Optimization Options

For faster processing, use these options:

```bash
# --fast: General speed optimizations
python spectrogram.py song.mp3 --fast

# --preview: Low DPI, fast rendering (72 DPI, 640x480 approx)
python spectrogram.py song.mp3 --preview

# --emergency: Maximum speed, sacrifices quality (sets 72 DPI, PDF format, bypasses matplotlib)
python spectrogram.py song.mp3 --emergency

# --bypass-matplotlib: Direct PIL image generation (5-20x faster than matplotlib)
python spectrogram.py song.mp3 --bypass-matplotlib

# Quality presets for automatic optimization
python spectrogram.py song.mp3 --quality draft    # Low quality, fast
python spectrogram.py song.mp3 --quality normal   # Balanced (default)
python spectrogram.py song.mp3 --quality high     # High quality, slower
```

### Output Format Options

```bash
# Output format selection (default: PNG)
python spectrogram.py song.mp3 --format png       # Highest quality, slower
python spectrogram.py song.mp3 --format pdf       # 5-10x faster than PNG
python spectrogram.py song.mp3 --format svg       # Vector format, scalable

# Image quality/DPI (affects file size and processing time)
python spectrogram.py song.mp3 --dpi 72           # Draft quality, very fast
python spectrogram.py song.mp3 --dpi 150          # Good quality, fast (recommended)
python spectrogram.py song.mp3 --dpi 300          # High quality, slow (default)
python spectrogram.py song.mp3 --dpi 600          # Very high quality, very slow
```

### Display & Visualization Options

```bash
# --log: Use logarithmic frequency scale (better for music analysis)
python spectrogram.py song.mp3 --log

# --no-display: Skip the window display after generation
python spectrogram.py song.mp3 --no-display

# --info: Display detailed audio file information
python spectrogram.py song.mp3 --info
```

### Advanced Options

```bash
# Threading control (0 = auto-detect, capped at 8 threads)
python spectrogram.py song.mp3 --threads 4

# Add annotations from JSON file
python spectrogram.py song.mp3 --annotations markers.json
```

## Usage Examples

### Quick Preview
Generate a quick, low-quality preview for inspection:
```bash
python spectrogram.py song.mp3 --preview --no-display
```

### High-Quality Analysis
Create high-quality spectrogram with logarithmic scale for music analysis:
```bash
python spectrogram.py song.flac --quality high --log
```

### Fast Processing of Large Files
Process large files quickly while maintaining reasonable quality:
```bash
python spectrogram.py long_audio.wav --fast --format pdf --dpi 150
```

### Emergency Mode
Get results immediately with maximum speed optimization:
```bash
python spectrogram.py audio.mp3 --emergency
```

### Batch Processing
Generate spectrograms for multiple files:
```bash
# Basic batch processing
for file in *.mp3; do
  python spectrogram.py "$file"
done

# With custom settings
for file in *.wav; do
  python spectrogram.py "$file" --fast --format pdf
done
```

## Supported Audio Formats

The tool supports any format that `soundfile` can read, including:

**Lossless Formats:**
- FLAC (.flac) - Free Lossless Audio Codec
- WAV (.wav) - PCM uncompressed
- AIFF (.aiff, .aif) - Audio Interchange File Format
- AU (.au) - Sun Audio Format
- CAF (.caf) - Core Audio Format
- APE (.ape) - Monkey's Audio
- WavPack (.wv) - WavPack Lossless
- And more...

**Lossy Formats:**
- MP3 (.mp3) - With ffmpeg backend
- AAC (.m4a) - With ffmpeg backend
- OGG Vorbis (.ogg) - With ffmpeg backend

## File Organization

The tool automatically organizes output into a structured folder:

```
spectro/
├── inputs/              # Place input audio files here (optional)
├── outputs/
│   ├── spectrograms/   # Generated spectrogram images
│   ├── performance_data/  # Performance metrics (learning data)
│   └── temp/           # Temporary files during processing
├── spectrogram.py      # Main script
└── spectrogram_usage.txt  # Quick reference guide
```

**Auto-generated filenames** are based on metadata:
- If metadata available: `Artist_-_Title.png`
- If no metadata: `filename.png`

**Output paths** automatically organized to `outputs/spectrograms/` folder.

## Performance Optimization

### Speed Comparison
- **Emergency mode**: ~0.5-2 seconds
- **Fast mode with PDF**: ~2-5 seconds
- **Normal PNG at 150 DPI**: ~5-15 seconds
- **High quality PNG at 300 DPI**: ~20-60 seconds

### Performance Tips

For large files (>50MB or >5 minutes):
```bash
# Use fast mode + PDF format + reduced DPI
python spectrogram.py large_file.wav --fast --format pdf --dpi 150
```

For maximum speed:
```bash
# Emergency mode provides 50x speedup
python spectrogram.py audio.mp3 --emergency
```

PDF format optimization:
```bash
# PDF is 5-10x faster than PNG
python spectrogram.py song.mp3 --format pdf
```

DPI impacts processing time:
```bash
# DPI 150 vs 300 = 3-5x speed difference
python spectrogram.py song.mp3 --dpi 150
```

### Performance Learning System

The tool maintains historical performance data to improve time estimates:
- **Load times** - Learns file reading performance
- **STFT computation** - Learns FFT processing speed with threading
- **Rendering times** - Learns format/quality specific timings

Data stored in: `outputs/performance_data/spectrogram_performance.pkl`

Estimates become more accurate as you process more files.

## Annotations

Add custom markers and labels to spectrograms using JSON annotation files.

### Annotation File Format

Create a `markers.json` file:
```json
[
  {
    "time": 25.5,
    "freq": 440,
    "label": "A4 Note",
    "color": "white",
    "marker": "o"
  },
  {
    "time": 45.2,
    "freq": 880,
    "label": "A5 Note",
    "color": "red",
    "marker": "s"
  }
]
```

### Usage
```bash
python spectrogram.py song.mp3 --annotations markers.json
```

**Annotation Fields:**
- `time` (float) - Time position in seconds
- `freq` (float) - Frequency position in Hz
- `label` (string) - Text label (optional)
- `color` (string) - Marker color (optional, default: "white")
- `marker` (string) - Marker style, one of: `o`, `s`, `^`, `v`, `*`, etc.

## Configuration

### Command-line Arguments Complete Reference

```
POSITIONAL:
  file_path              Path to input audio file

OUTPUT:
  -o, --output           Output filename (default: auto-generated from metadata)
  --format {png,pdf,svg} Output format (default: png, pdf is 5-10x faster)
  --dpi DPI              Output DPI (default: 300, try 150 for faster)

SPEED & QUALITY:
  --fast                 Fast mode with optimizations for speed
  --preview              Preview mode (low DPI 72, fast rendering)
  --emergency            Emergency speed mode (50x faster, lower quality)
  --quality {draft,normal,high}  Quality preset (default: normal)
  --bypass-matplotlib    Use direct PIL generation (5-20x faster, basic output)

VISUALIZATION:
  --log                  Use logarithmic frequency axis (better for music)
  --no-display           Don't show the plot window after generation

INFORMATION:
  --info                 Display detailed audio file information
  --hotkeys              Show all available commands and options

THREADING:
  --threads N            Number of threads (0 = auto-detect, capped at 8)

ANNOTATIONS:
  --annotations FILE     JSON file with annotations/markers for spectrogram

EXAMPLES:
  python spectrogram.py song.mp3
  python spectrogram.py song.flac -o custom_name.png
  python spectrogram.py song.mp3 --preview --no-display
  python spectrogram.py song.wav --quality high --log
  python spectrogram.py audio.mp3 --emergency
```

## Performance Metrics & Troubleshooting

### Large File Performance Hints

```
File > 50MB detected:
Consider using --preview or --fast for quicker processing
```

### Slow Save Performance

If saving takes more than 30 seconds:
```bash
# For next run, try these optimizations:
python spectrogram.py audio.mp3 --emergency
python spectrogram.py audio.mp3 --bypass-matplotlib
python spectrogram.py audio.mp3 --format pdf
python spectrogram.py audio.mp3 --dpi 150
```

### Special Cases

**Emergency Save Timeout:**
- If matplotlib save exceeds 30 seconds, automatically switches to direct PIL save
- Creates rescue file: `filename_rescue.png`
- Provides acceptable quality at 72 DPI

## Audio Information Display

Use `--info` flag to see detailed audio file analysis:

```bash
python spectrogram.py audio.wav --info
```

Displays:
- **File**: Name and full path
- **Size**: File size in MB and bytes
- **Format**: WAV, FLAC, MP3, etc.
- **Compression**: Type of codec/compression
- **Lossless**: Yes/No
- **Sample Rate**: Frequency (Hz)
- **Bit Depth**: 8, 16, 24, or 32 bits
- **Channels**: Mono, Stereo, or multi-channel
- **Bitrate**: Calculated bitrate (kbps)
- **Duration**: Total duration in MM:SS.ms
- **Total Frames**: Number of audio samples

## Development & Architecture

### Key Components

1. **STFT Computation** - Multi-threaded FFT using numpy with configurable window sizes
2. **Direct Rendering** - PIL-based image generation for speed
3. **Matplotlib Rendering** - High-quality matplotlib backend for standard output
4. **Performance Tracking** - Historical data collection for intelligent time estimation
5. **File Organization** - Automatic folder structure management
6. **Metadata Extraction** - Using mutagen for tag reading

### Script Structure

- **Lines 1-87**: Imports and argument parsing
- **Lines 197-207**: File validation
- **Lines 219-295**: Audio file information extraction
- **Lines 340-359**: Optimal STFT parameter calculation
- **Lines 382-426**: Multi-threaded STFT computation
- **Lines 428-479**: Intelligent time estimation
- **Lines 481-553**: Direct PIL-based image generation
- **Lines 556-641**: Audio loading and STFT processing
- **Lines 658-738**: Spectrogram visualization and plotting
- **Lines 751-804**: Image saving with timeout protection

## Troubleshooting

### No audio files showing
Ensure audio file path is correct and format is supported by soundfile:
```bash
python spectrogram.py --info  # Displays info without generating spectrogram
```

### Slow processing
Use performance optimization options:
```bash
python spectrogram.py audio.mp3 --fast --format pdf --dpi 150
```

### Missing output file
Check `outputs/spectrograms/` folder - files are automatically organized there:
```bash
ls -lh outputs/spectrograms/
```

### Metadata not found
If metadata extraction fails, filenames fall back to original filename:
- Ensure mutagen is installed: `pip install mutagen`
- Check audio file contains proper ID3 tags (for MP3) or Vorbis comments (for OGG)

### Display window won't open
Use `--no-display` flag and check saved image file directly:
```bash
python spectrogram.py audio.mp3 --no-display
```

## Advanced Usage

### Batch Processing with Status Tracking

```bash
#!/bin/bash
# Process all MP3 files with fast settings
for file in *.mp3; do
  echo "Processing: $file"
  python spectrogram.py "$file" --fast --format pdf --dpi 150
done

echo "Done! Check outputs/spectrograms/"
```

### Custom Annotation Workflow

```bash
# Generate base spectrogram
python spectrogram.py song.mp3 -o base.png

# Create markers.json with time/frequency annotations
# Then regenerate with annotations
python spectrogram.py song.mp3 --annotations markers.json
```

## Output Examples

The tool generates spectrograms with:
- **Inferno colormap** - Dark purple to bright yellow/white color scheme
- **Time axis** - Horizontal, in seconds
- **Frequency axis** - Vertical, in Hz or log scale
- **Colorbar** - Intensity scale in dB
- **Title** - Audio filename
- **Metadata** - Song metadata if available in the audio file

Example outputs are stored in:
```
outputs/spectrograms/
├── Bob_Dylan_-_Visions_of_Johanna_Album_Version_direct.png
├── Bob_Dylan_Like_a_Rolling_Stone.png
└── Sakamoto_rescue.png
```

## System Notes

- **Virtual environment**: Uses Python 3.13.5 from Homebrew
- **CPU cores detected**: Automatically uses optimal thread count (capped at 8)
- **Memory efficient**: Uses float32 for STFT to minimize memory usage
- **Compatibility**: Tested on macOS (Darwin 25.1.0)

## License

See LICENSE file for licensing information.

## Support & Questions

For detailed command information:
```bash
python spectrogram.py --hotkeys
```

For step-by-step processing:
```bash
python spectrogram.py --info
```

For performance data analysis:
```bash
ls -lh outputs/performance_data/
```

---

**Last Updated**: November 2024
**Version**: 3.0+ (Minimal Aesthetic UI with Intelligent Time Estimation)
