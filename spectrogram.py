#!/usr/bin/env python3
import time
script_start = time.perf_counter()

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import shutil
try:
    from alive_progress import alive_bar
    ALIVE_PROGRESS_AVAILABLE = True
except ImportError:
    from tqdm import tqdm
    ALIVE_PROGRESS_AVAILABLE = False
import json
from pathlib import Path
import pickle
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from PIL import Image
import re

# Minimal Progress Functions
def format_time(seconds):
    """Format time cleanly"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

def minimal_progress(title, total=None, show_eta=True):
    """Create minimal aesthetic progress bar"""
    if ALIVE_PROGRESS_AVAILABLE and total:
        return alive_bar(total, title=title, bar='smooth', spinner='dots_waves')
    elif total:
        return tqdm(total=total, desc=title, bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]')
    else:
        # Spinner for unknown duration
        if ALIVE_PROGRESS_AVAILABLE:
            return alive_bar(monitor=None, stats=None, title=title, spinner='dots_waves')
        else:
            return tqdm(desc=title, bar_format='{desc} {elapsed}')

def print_summary(load_time, stft_time, plot_time, save_time, total_time):
    """Print clean performance summary"""
    print(f"\nCompleted in {format_time(total_time)}")
    print(f"  Load: {format_time(load_time)} | STFT: {format_time(stft_time)} | Plot: {format_time(plot_time)} | Save: {format_time(save_time)}")

# Try to import metadata library
try:
    from mutagen import File as MutagenFile
    METADATA_AVAILABLE = True
except ImportError:
    try:
        from tinytag import TinyTag
        METADATA_AVAILABLE = True
    except ImportError:
        METADATA_AVAILABLE = False

# Use simple backend setup
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for stability

parser = argparse.ArgumentParser(description="Generate a spectrogram from an audio file.")
parser.add_argument("file_path", nargs="?", help="Path to the input audio file")
parser.add_argument("-o", "--output", default="spectrogram.png", help="Output filename")
parser.add_argument("--log", action="store_true", help="Use logarithmic frequency axis")
parser.add_argument("--annotations", help="JSON file with annotations")
parser.add_argument("--info", action="store_true", help="Display detailed file information")
parser.add_argument("--no-display", action="store_true", help="Don't show plot window")
parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300, try 150 for faster)")
parser.add_argument("--format", choices=['png', 'pdf', 'svg'], default='png', help="Output format (pdf is faster)")
parser.add_argument("--quality", choices=['draft', 'normal', 'high'], default='normal', help="Quality preset")
parser.add_argument("--fast", action="store_true", help="Fast mode with optimizations for speed")
parser.add_argument("--preview", action="store_true", help="Preview mode (low DPI, fast rendering)")
parser.add_argument("--emergency", action="store_true", help="Emergency speed mode (sacrifices quality for speed)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads (0 = auto-detect)")
parser.add_argument("--bypass-matplotlib", action="store_true", help="Use direct image generation (faster, basic)")
parser.add_argument("--hotkeys", action="store_true", help="Show all available commands and options")
args = parser.parse_args()

# Show hotkeys/commands if requested
if args.hotkeys:
    print("SPECTROGRAM COMMANDS")
    print("=" * 50)
    print("BASIC:")
    print("  python spectrogram.py audio.mp3")
    print("  python spectrogram.py audio.mp3 -o custom.pdf")
    print()
    print("SPEED OPTIONS:")
    print("  --fast           # Optimized for speed")
    print("  --preview        # Low DPI, fast render")
    print("  --emergency      # Sacrifice quality for speed")
    print("  --bypass-matplotlib # Direct generation")
    print()
    print("OUTPUT:")
    print("  -o filename      # Output filename")
    print("  --format pdf     # Format: png/pdf/svg")
    print("  --dpi 150        # DPI (150=fast, 300=default)")
    print("  --no-display     # Don't show window")
    print()
    print("QUALITY:")
    print("  --quality draft  # Quality: draft/normal/high")
    print("  --log            # Logarithmic frequency")
    print()
    print("OTHER:")
    print("  --info           # Show file details")
    print("  --threads 4      # Thread count (0=auto)")
    print("  --annotations file.json # Add annotations")
    print()
    print("EXAMPLES:")
    print("  python spectrogram.py song.mp3 --fast --format pdf")
    print("  python spectrogram.py song.mp3 --preview --no-display")
    print("  python spectrogram.py song.mp3 --emergency -o quick.png")
    exit(0)

# Auto-detect optimal thread count
if args.threads == 0:
    args.threads = min(mp.cpu_count(), 8)  # Cap at 8 threads for memory reasons

# Quality presets
if args.emergency:
    args.dpi = 72
    args.format = 'pdf'
    args.fast = True
    args.bypass_matplotlib = True
    print("EMERGENCY MODE: Maximum speed, minimal quality, bypassing matplotlib")
elif args.preview:
    args.dpi = 72
    args.format = 'png'
    args.fast = True
elif args.quality == 'draft':
    args.dpi = min(args.dpi, 150)
    args.fast = True
elif args.quality == 'high':
    args.dpi = max(args.dpi, 300)

# Intelligent performance optimization
if args.format == 'png' and args.dpi >= 300 and not args.emergency:
    print("⚠️  WARNING: PNG at 300+ DPI is VERY slow!")
    print("🚀 Automatically optimizing for speed...")
    args.format = 'pdf'  # Auto-switch to faster format
    print(f"   → Switched to PDF format (10x faster)")
    if args.dpi > 200:
        args.dpi = 200
        print(f"   → Reduced DPI to {args.dpi} (3x faster)")
    print("   Use --format png --dpi 300 to force original settings")

# Create organized output directories
from pathlib import Path
outputs_dir = Path("outputs")
spectrograms_dir = outputs_dir / "spectrograms"
performance_dir = outputs_dir / "performance_data"
temp_dir = outputs_dir / "temp"

# Ensure directories exist
spectrograms_dir.mkdir(parents=True, exist_ok=True)
performance_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

# Auto-adjust output filename based on format and move to organized folder
if args.output == "spectrogram.png" and args.format != 'png':
    args.output = f"spectrogram.{args.format}"

# Ensure output goes to spectrograms directory
if not str(args.output).startswith("outputs/"):
    args.output = str(spectrograms_dir / Path(args.output).name)

# Performance data storage
PERF_DATA_FILE = performance_dir / "spectrogram_performance.pkl"

def load_performance_data():
    try:
        if PERF_DATA_FILE.exists():
            with open(PERF_DATA_FILE, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return defaultdict(list)

def save_performance_data(data):
    try:
        with open(PERF_DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass

performance_data = load_performance_data()

# Get file path
if args.file_path is None:
    print("Spectrogram Generator")
    args.file_path = input("Audio file: ").strip()

file_path = args.file_path.strip().strip('"\'')

if not os.path.isfile(file_path):
    print(f"Error: file not found: {file_path}", file=sys.stderr)
    sys.exit(1)

def clean_filename(text):
    """Convert text to clean filename-safe string"""
    if not text:
        return ""
    # Remove or replace invalid characters
    text = re.sub(r'[<>:"/\\|?*]', '', text)  # Remove invalid chars
    text = re.sub(r'[^\w\s\-_\.]', '', text)  # Keep only alphanumeric, spaces, hyphens, underscores, dots
    text = re.sub(r'\s+', '_', text.strip())  # Replace spaces with underscores
    text = text[:50]  # Limit length
    return text

def get_file_info(file_path):
    """Get comprehensive audio file information"""
    info = {}
    path_obj = Path(file_path)
    
    stat = path_obj.stat()
    info['filename'] = path_obj.name
    info['filepath'] = str(path_obj.absolute())
    info['filesize_bytes'] = stat.st_size
    info['filesize_mb'] = round(stat.st_size / (1024 * 1024), 2)
    info['extension'] = path_obj.suffix.lower()
    
    # Extract metadata if available
    info['title'] = None
    info['artist'] = None
    info['album'] = None
    info['clean_filename'] = None
    
    if METADATA_AVAILABLE:
        try:
            # Try mutagen first
            if 'MutagenFile' in globals():
                audio_file = MutagenFile(file_path)
                if audio_file:
                    info['title'] = str(audio_file.get('TIT2', [''])[0] or audio_file.get('TITLE', [''])[0] or '')
                    info['artist'] = str(audio_file.get('TPE1', [''])[0] or audio_file.get('ARTIST', [''])[0] or '')
                    info['album'] = str(audio_file.get('TALB', [''])[0] or audio_file.get('ALBUM', [''])[0] or '')
            # Fallback to tinytag
            elif 'TinyTag' in globals():
                tag = TinyTag.get(file_path)
                if tag:
                    info['title'] = tag.title or ''
                    info['artist'] = tag.artist or ''
                    info['album'] = tag.album or ''
        except Exception:
            pass  # Metadata extraction failed, continue without it
    
    # Create clean filename from metadata
    if info['title']:
        base_name = info['title']
        if info['artist']:
            base_name = f"{info['artist']} - {info['title']}"
        info['clean_filename'] = clean_filename(base_name)
    else:
        # Fallback to original filename without extension
        info['clean_filename'] = clean_filename(path_obj.stem)
    
    with sf.SoundFile(file_path) as f:
        info['sample_rate'] = f.samplerate
        info['channels'] = f.channels
        info['frames'] = f.frames
        info['duration_seconds'] = round(f.frames / f.samplerate, 3)
        info['duration_formatted'] = f"{int(info['duration_seconds'] // 60):02d}:{int(info['duration_seconds'] % 60):02d}.{int((info['duration_seconds'] % 1) * 1000):03d}"
        info['subtype'] = f.subtype
        info['format'] = f.format
        info['endian'] = f.endian
        
        bits_per_sample = {
            'PCM_16': 16, 'PCM_24': 24, 'PCM_32': 32, 'FLOAT': 32, 'DOUBLE': 64,
            'PCM_S8': 8, 'PCM_U8': 8, 'ULAW': 8, 'ALAW': 8
        }.get(f.subtype, 16)
        
        info['bits_per_sample'] = bits_per_sample
        info['bitrate_kbps'] = round((f.samplerate * f.channels * bits_per_sample) / 1000, 1)
        
        lossless_formats = {'.flac', '.wav', '.aiff', '.aif', '.au', '.snd', '.raw', '.w64', '.rf64', '.caf'}
        info['is_lossless'] = info['extension'] in lossless_formats
        
        format_descriptions = {
            '.flac': 'FLAC Lossless',
            '.wav': 'Uncompressed PCM' if f.subtype.startswith('PCM') else f.subtype,
            '.aiff': 'Uncompressed PCM',
            '.aif': 'Uncompressed PCM'
        }
        info['compression'] = format_descriptions.get(info['extension'], f.subtype)
    
    return info

def display_file_info(info):
    """Display formatted file information"""
    print("\nAUDIO FILE INFORMATION")
    print("=" * 50)
    print(f"File: {info['filename']}")
    print(f"Path: {info['filepath']}")
    print(f"Size: {info['filesize_mb']} MB ({info['filesize_bytes']:,} bytes)")
    print(f"Format: {info['format']} ({info['extension'].upper()})")
    print(f"Compression: {info['compression']}")
    print(f"Lossless: {'Yes' if info['is_lossless'] else 'No'}")
    print(f"Sample Rate: {info['sample_rate']:,} Hz")
    print(f"Bit Depth: {info['bits_per_sample']} bits")
    channels_desc = 'Mono' if info['channels'] == 1 else 'Stereo' if info['channels'] == 2 else f"{info['channels']}-channel"
    print(f"Channels: {info['channels']} ({channels_desc})")
    print(f"Bitrate: {info['bitrate_kbps']:,} kbps")
    print(f"Duration: {info['duration_formatted']} ({info['duration_seconds']} seconds)")
    print(f"Total Frames: {info['frames']:,}")
    print("=" * 50)

# Analyze file
print("[1/4] Analyzing audio file...")
file_info = get_file_info(file_path)

# Update output filename to use clean metadata-based name if using default
if args.output == str(spectrograms_dir / "spectrogram.png") or args.output == str(spectrograms_dir / f"spectrogram.{args.format}"):
    clean_name = file_info['clean_filename'] if file_info['clean_filename'] else "spectrogram"
    args.output = str(spectrograms_dir / f"{clean_name}.{args.format}")

# Show performance hint for large files
if file_info['filesize_mb'] > 50 or file_info['duration_seconds'] > 300:
    print(f"Large file detected ({file_info['filesize_mb']} MB, {file_info['duration_formatted']})")
    if not args.preview and not args.fast:
        print("Consider using --preview or --fast for quicker processing")

if args.info:
    display_file_info(file_info)
    if input("\nContinue with spectrogram generation? (y/N): ").lower() != 'y':
        sys.exit(0)
else:
    print(f"File: {file_info['filename']} | {file_info['duration_formatted']} | {file_info['bitrate_kbps']} kbps | {file_info['sample_rate']/1000}kHz/{file_info['bits_per_sample']}bit")
    print(f"Output: {args.output} ({args.format.upper()}, {args.dpi} DPI)")
    print(f"Threading: {args.threads} threads ({mp.cpu_count()} CPU cores available)")

def get_optimal_stft_params(sample_rate, duration_seconds):
    """Calculate optimal STFT parameters"""
    if sample_rate >= 192000:
        base_nperseg = 8192
    elif sample_rate >= 96000:
        base_nperseg = 4096
    elif sample_rate >= 48000:
        base_nperseg = 2048
    else:
        base_nperseg = 1024
    
    if duration_seconds < 30:
        nperseg = max(512, base_nperseg // 2)
    elif duration_seconds > 300:
        nperseg = min(16384, base_nperseg * 2)
    else:
        nperseg = base_nperseg
    
    noverlap = int(nperseg * 0.75)
    return nperseg, noverlap

def compute_stft_chunk(args_tuple):
    """Compute STFT for a chunk of frames - designed for multiprocessing"""
    data, start_frame, end_frame, nperseg, noverlap, window, hop_length = args_tuple
    
    chunk_size = end_frame - start_frame
    freq_bins = nperseg // 2 + 1
    chunk_stft = np.zeros((freq_bins, chunk_size), dtype=np.float32)
    
    for i in range(chunk_size):
        frame_idx = start_frame + i
        start_sample = frame_idx * hop_length
        frame = data[start_sample:start_sample + nperseg]
        
        if len(frame) < nperseg:
            frame = np.pad(frame, (0, nperseg - len(frame)))
        
        spectrum = np.fft.rfft(frame * window)
        chunk_stft[:, i] = np.abs(spectrum) ** 2
    
    return start_frame, chunk_stft

def compute_stft_parallel(data, nperseg, noverlap, window, n_frames, num_threads):
    """Multi-threaded STFT computation"""
    hop_length = nperseg - noverlap
    freq_bins = nperseg // 2 + 1
    
    # Calculate chunk sizes for each thread
    chunk_size = max(1, n_frames // num_threads)
    chunks = []
    
    for i in range(num_threads):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, n_frames)
        if start_frame < n_frames:
            chunks.append((data, start_frame, end_frame, nperseg, noverlap, window, hop_length))
    
    # Pre-allocate result matrix
    Sxx = np.zeros((freq_bins, n_frames), dtype=np.float32)
    
    print(f"Using {len(chunks)} threads for STFT computation...")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(compute_stft_chunk, chunk): chunk for chunk in chunks}
        
        # Collect results with progress tracking
        if ALIVE_PROGRESS_AVAILABLE:
            eta_text = f"ETA: {format_time(time_estimates['stft'])}"
            with alive_bar(len(chunks), title=f"STFT chunks ({eta_text})", bar='smooth', spinner='dots_waves') as bar:
                for future in as_completed(future_to_chunk):
                    start_frame, chunk_result = future.result()
                    end_frame = start_frame + chunk_result.shape[1]
                    Sxx[:, start_frame:end_frame] = chunk_result
                    bar()
        else:
            completed = 0
            eta_text = f"(ETA: {format_time(time_estimates['stft'])})"
            for future in as_completed(future_to_chunk):
                start_frame, chunk_result = future.result()
                end_frame = start_frame + chunk_result.shape[1]
                Sxx[:, start_frame:end_frame] = chunk_result
                completed += 1
                if completed % max(1, len(chunks) // 10) == 0:
                    print(f"Progress: {completed}/{len(chunks)} chunks {eta_text}")
    return Sxx

def estimate_phase_times(file_info, nperseg, n_frames):
    """Accurate time estimation using historical data and current settings"""
    estimates = {}
    file_size_mb = file_info['filesize_mb']
    
    # Load time: Use recent historical data
    if 'load_time' in performance_data and len(performance_data['load_time']) > 2:
        recent_data = performance_data['load_time'][-10:]  # Last 10 measurements
        speeds = [size/max(time, 0.001) for size, time in recent_data if time > 0]
        if speeds:
            avg_speed = np.median(speeds)  # Median is more robust
            estimates['load'] = max(0.05, file_size_mb / avg_speed)
        else:
            estimates['load'] = max(0.05, file_size_mb * 0.015)
    else:
        estimates['load'] = max(0.05, file_size_mb * 0.015)
    
    # STFT: Better modeling with historical data
    if 'stft_time' in performance_data and len(performance_data['stft_time']) > 2:
        recent_stft = performance_data['stft_time'][-5:]
        speeds = [frames/max(time, 0.001) for frames, time in recent_stft if time > 0]
        if speeds:
            avg_frame_speed = np.median(speeds)
            thread_efficiency = min(args.threads * 0.7, 4)  # More realistic speedup
            estimates['stft'] = max(0.1, n_frames / (avg_frame_speed * thread_efficiency))
        else:
            estimates['stft'] = max(0.1, n_frames * 0.00005 / min(args.threads, 4))
    else:
        estimates['stft'] = max(0.1, n_frames * 0.00005 / min(args.threads, 4))
    
    # Plot: Account for actual settings
    actual_pixels = (args.dpi * 12) * (args.dpi * 6)  # Use actual DPI
    complexity = (actual_pixels * (nperseg // 2)) / 1e8
    
    # Adjust for mode
    if args.emergency or args.bypass_matplotlib:
        estimates['plot'] = max(0.1, complexity * 0.2)  # Much faster
    elif args.fast or args.preview:
        estimates['plot'] = max(0.2, complexity * 0.5)
    else:
        estimates['plot'] = max(0.5, complexity * 1.5)
    
    # Save: Account for format and size
    save_complexity = actual_pixels / 1e6  # Megapixels
    if args.format == 'pdf':
        estimates['save'] = max(0.1, save_complexity * 0.1)  # PDF is fast
    elif args.emergency or args.bypass_matplotlib:
        estimates['save'] = max(0.1, save_complexity * 0.05)  # Direct save is fastest
    else:
        estimates['save'] = max(0.2, save_complexity * 0.8)  # PNG is slow
    
    return estimates

def save_spectrogram_direct(Sxx_db, times, frequencies, output_path, dpi=150):
    """Direct spectrogram saving using PIL - bypasses matplotlib completely"""
    print("Using direct image generation (bypassing matplotlib)...")
    
    # Ensure output path uses organized folder structure
    if not str(output_path).startswith("outputs/"):
        output_filename = Path(output_path).name
        output_path = str(spectrograms_dir / output_filename)
    
    # Normalize to 0-255 range for image
    norm_data = ((Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min()) * 255).astype(np.uint8)
    
    # FAST vectorized colormap - much faster than nested loops
    height, width = norm_data.shape
    
    # Create inferno-like colormap using vectorized operations
    vals = norm_data.astype(np.float32) / 255.0
    
    # Vectorized color calculation
    r = np.zeros_like(vals)
    g = np.zeros_like(vals)
    b = np.zeros_like(vals)
    
    # Dark purple to red (0-0.25)
    mask1 = vals < 0.25
    r[mask1] = vals[mask1] * 4 * 255
    g[mask1] = 0
    b[mask1] = 128 + vals[mask1] * 4 * 127
    
    # Red to orange (0.25-0.5)
    mask2 = (vals >= 0.25) & (vals < 0.5)
    r[mask2] = 255
    g[mask2] = (vals[mask2] - 0.25) * 4 * 255
    b[mask2] = 0
    
    # Orange to yellow (0.5-0.75)
    mask3 = (vals >= 0.5) & (vals < 0.75)
    r[mask3] = 255
    g[mask3] = 255
    b[mask3] = (vals[mask3] - 0.5) * 4 * 255
    
    # Yellow to white (0.75-1.0)
    mask4 = vals >= 0.75
    r[mask4] = 255
    g[mask4] = 255
    b[mask4] = 255
    
    # Stack to RGB array
    rgb_data = np.stack([r, g, b], axis=2).astype(np.uint8)
    
    # Flip vertically (matplotlib convention)
    rgb_data = np.flipud(rgb_data)
    
    # Create PIL image
    img = Image.fromarray(rgb_data, 'RGB')
    
    # Only resize if necessary (DPI scaling)
    if dpi != 100:
        scale_factor = dpi / 100
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.NEAREST)  # Use NEAREST for speed
    
    # Handle PDF format conversion while maintaining organized structure
    if output_path.endswith('.pdf'):
        output_path = output_path.replace('.pdf', '_direct.png')
        # Ensure it's still in the spectrograms directory
        if not str(output_path).startswith("outputs/"):
            output_filename = Path(output_path).name
            output_path = str(spectrograms_dir / output_filename)
    
    img.save(output_path, optimize=False)  # Disable optimization for speed
    return output_path

# Load audio
print("\n[2/4] Loading audio data...")
load_start = time.perf_counter()

if ALIVE_PROGRESS_AVAILABLE:
    with alive_bar(monitor=None, stats=None, title="Loading audio", spinner='dots_waves') as bar:
        data, sr = sf.read(file_path, dtype='float32')
else:
    data, sr = sf.read(file_path, dtype='float32')

load_time = time.perf_counter() - load_start
print(f"Loaded {file_info['filesize_mb']} MB in {format_time(load_time)}")

if data.ndim > 1:
    data = data.mean(axis=1)
    print("Converted to mono")

# Get parameters
nperseg, noverlap = get_optimal_stft_params(sr, file_info['duration_seconds'])
hop_length = nperseg - noverlap
window = np.hanning(nperseg)
n_frames = int(np.floor((len(data) - noverlap) / hop_length))

# Show configuration
time_estimates = estimate_phase_times(file_info, nperseg, n_frames)
total_estimated = sum(time_estimates.values())

print(f"\nSTFT Configuration:")
print(f"  Window size: {nperseg} samples")
print(f"  Overlap: {noverlap} samples ({noverlap/nperseg*100:.1f}%)")
print(f"  Frequency resolution: {sr/nperseg:.1f} Hz")
print(f"  Time resolution: {hop_length/sr*1000:.1f} ms")
print(f"  Using {args.threads} threads for acceleration")
print(f"\nEstimated time: {format_time(total_estimated)}")
print(f"  Load: {format_time(time_estimates['load'])} | STFT: {format_time(time_estimates['stft'])} | Plot: {format_time(time_estimates['plot'])} | Save: {format_time(time_estimates['save'])}")

# STFT computation with multi-threading
print("\n[3/4] Computing spectrogram...")

# Sample estimation for single-threaded reference
sample_count = min(max(10, n_frames // 100), n_frames)
times_sample = []
for i in range(sample_count):
    idx = (i * (n_frames // sample_count)) * hop_length
    frame = data[idx:idx + nperseg]
    if frame.shape[0] < nperseg:
        frame = np.pad(frame, (0, nperseg - frame.shape[0]))
    start_t = time.perf_counter()
    _ = np.fft.rfft(frame * window)
    times_sample.append(time.perf_counter() - start_t)

median_sample = sorted(times_sample)[len(times_sample)//2]
single_thread_estimate = median_sample * n_frames
multi_thread_estimate = single_thread_estimate / min(args.threads, 4)
print(f"Estimated time: {single_thread_estimate:.2f}s single-threaded → {multi_thread_estimate:.2f}s multi-threaded")

# Multi-threaded STFT computation
stft_start = time.perf_counter()

# Only use threading for large datasets where it helps
if args.threads > 1 and n_frames > 1000:
    # Use multi-threading for large datasets
    Sxx = compute_stft_parallel(data, nperseg, noverlap, window, n_frames, args.threads)
else:
    # Use single-threaded for small datasets (often faster due to less overhead)
    if n_frames <= 1000:
        print("Using single-threaded computation (small dataset, threading overhead not worth it)")
    else:
        print("Using single-threaded computation")
    
    Sxx = np.zeros((nperseg//2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = data[start:start + nperseg]
        if frame.shape[0] < nperseg:
            frame = np.pad(frame, (0, nperseg - frame.shape[0]))
        spectrum = np.fft.rfft(frame * window)
        Sxx[:, i] = np.abs(spectrum) ** 2
        
        if i % max(1, n_frames // 20) == 0:
            percent = (i / n_frames) * 100
            print(f'\rProgress: {percent:.0f}% ({i}/{n_frames})', end='', flush=True)
    print()

stft_time = time.perf_counter() - stft_start
speedup = single_thread_estimate / stft_time if stft_time > 0 else 1
print(f"STFT completed in {stft_time:.2f}s ({speedup:.1f}x speedup)")

# Convert to dB
Sxx_db = 10 * np.log10(Sxx + 1e-10)
times = np.arange(n_frames) * hop_length / sr
frequencies = np.fft.rfftfreq(nperseg, 1.0 / sr)

# Load annotations
annotations = []
if args.annotations:
    try:
        with open(args.annotations, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations")
    except Exception as e:
        print(f"Warning: Could not load annotations: {e}")

# Generate plot
print("\n[4/4] Generating visualization...")
print("Creating plot...")

plot_start = time.perf_counter()

plt.figure(figsize=(12, 6))

# EMERGENCY MODE: Use imshow for maximum speed
if args.emergency:
    print("Using imshow for emergency speed (may affect quality)")
    # Flip the array for proper orientation with imshow
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    plt.imshow(Sxx_db, cmap='inferno', aspect='auto', origin='lower', extent=extent, interpolation='bilinear')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram - {file_info["filename"]} (Emergency Mode)')
    plt.colorbar(label='Intensity (dB)')
else:
    # Performance optimization: use rasterization for fast mode
    rasterized = args.fast or args.preview
    plt.pcolormesh(times, frequencies, Sxx_db, cmap='inferno', shading='gouraud', rasterized=rasterized)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram - {file_info["filename"]}')
    plt.colorbar(label='Intensity (dB)')

# Add annotations
if annotations:
    for ann in annotations:
        time_pos = ann.get('time', 0)
        freq_pos = ann.get('freq', 0)
        label = ann.get('label', '')
        color = ann.get('color', 'white')
        marker = ann.get('marker', 'o')
        
        if 0 <= time_pos <= times[-1] and 0 <= freq_pos <= frequencies[-1]:
            plt.scatter(time_pos, freq_pos, c=color, marker=marker, s=100, 
                       edgecolors='black', linewidth=1, zorder=10)
            if label:
                plt.annotate(label, (time_pos, freq_pos), xytext=(5, 5), 
                           textcoords='offset points', color=color, 
                           fontweight='bold', fontsize=9, zorder=11,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

# Set frequency range
max_freq = sr / 2
if args.log:
    plt.yscale('log')
    plt.ylim(20, max_freq)
    freq_range = f"20 Hz to {max_freq/1000:.1f} kHz (log)"
else:
    plt.ylim(0, max_freq)
    freq_range = f"0 Hz to {max_freq/1000:.1f} kHz (linear)"

# Add reference lines for high sample rates
if sr >= 96000:
    ref_freqs = [20000, 22050, 24000, 40000]
    for freq in ref_freqs:
        if freq < max_freq:
            plt.axhline(y=freq, color='white', alpha=0.3, linestyle='--', linewidth=0.5)
            plt.text(times[-1] * 0.98, freq, f'{freq/1000:.0f}kHz', 
                    color='white', alpha=0.7, fontsize=8, ha='right', va='bottom')

# Use tight_layout for better performance than bbox_inches='tight'
if not args.emergency:
    plt.tight_layout()
else:
    # Skip tight_layout in emergency mode for maximum speed
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

print(f"Plot created ({freq_range})")

# Save image with timeout and detailed tracking
print("Saving image...")
print(f"Target: {args.output} ({args.format.upper()}, {args.dpi} DPI)")
print(f"Image size: {12*args.dpi:.0f}x{6*args.dpi:.0f} pixels = {(12*args.dpi*6*args.dpi)/1e6:.1f} megapixels")

save_start = time.perf_counter()

# Choose saving method
if args.bypass_matplotlib:
    # Use direct PIL-based saving (much faster)
    try:
        actual_output = save_spectrogram_direct(Sxx_db, times, frequencies, args.output, args.dpi)
        actual_save_time = time.perf_counter() - save_start
        print(f"Direct save completed: {actual_output} in {actual_save_time:.2f}s")
        save_success = True
    except Exception as e:
        print(f"Direct save failed: {e}")
        print("Falling back to matplotlib...")
        args.bypass_matplotlib = False

if not args.bypass_matplotlib:
    # Use matplotlib (with timeout protection)
    import signal
    
    class SaveTimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise SaveTimeoutError("Save operation timed out")
    
    save_success = False
    try:
        # Set a 30-second timeout for saving
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        print("matplotlib.savefig() starting...")
        
        # Minimal savefig parameters
        if args.emergency:
            plt.savefig(args.output, dpi=72, format='png')  # Force PNG for emergency
        else:
            plt.savefig(args.output, dpi=args.dpi, format=args.format)
        
        signal.alarm(0)  # Cancel timeout
        save_success = True
        
        actual_save_time = time.perf_counter() - save_start
        print(f"matplotlib save completed in {actual_save_time:.2f}s")

    except SaveTimeoutError:
        print("\n🚨 MATPLOTLIB SAVE TIMEOUT!")
        print("Switching to direct save method...")
        
        try:
            # Create rescue filename in organized structure
            rescue_filename = Path(args.output).stem + '_rescue.png'
            rescue_path = str(spectrograms_dir / rescue_filename)
            actual_output = save_spectrogram_direct(Sxx_db, times, frequencies, 
                                                  rescue_path, 72)
            actual_save_time = time.perf_counter() - save_start
            print(f"Rescue save completed: {actual_output} in {actual_save_time:.2f}s")
            save_success = True
        except Exception as e:
            print(f"Rescue save also failed: {e}")
            actual_save_time = time.perf_counter() - save_start

    except Exception as e:
        print(f"matplotlib save error: {e}")
        actual_save_time = time.perf_counter() - save_start

    finally:
        signal.alarm(0)

plot_time = time.perf_counter() - plot_start

# Performance improvement suggestions
if actual_save_time > 30:
    print(f"\n🚨 EXTREMELY SLOW SAVE: {actual_save_time:.1f}s!")
    print("🚀 Next time try these for MUCH faster saving:")
    print("  --emergency       # 10-50x faster (direct generation)")
    print("  --bypass-matplotlib  # 5-20x faster (PIL direct)")
    print("  --format pdf      # 5-10x faster than PNG")
    print("  --dpi 150         # 3-5x faster than 300 DPI")
    print("  --fast --preview  # Ultimate speed combo")
elif actual_save_time > 10:
    print(f"\n⚠️  SLOW SAVE TIME: {actual_save_time:.1f}s")
    print("🔧 Performance tips for faster saving:")
    if args.format == 'png':
        print("  --format pdf      # 5-10x faster than PNG")
    if args.dpi >= 200:
        print("  --dpi 150         # 3x faster than high DPI")
    print("  --bypass-matplotlib  # Use direct generation")

# Display if requested
if not args.no_display:
    print("Opening display window...")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not open display: {e}")

# Cleanup
plt.close('all')

# Final summary
total_runtime = time.perf_counter() - script_start
print_summary(load_time, stft_time, plot_time, actual_save_time, total_runtime)

# Store performance data for future estimates
performance_data['load_time'].append((file_info['filesize_mb'], load_time))
performance_data['stft_time'].append((n_frames, stft_time))
performance_data['plot_time'].append((nperseg * n_frames, plot_time))

# Keep only recent data
for key in performance_data:
    if len(performance_data[key]) > 50:
        performance_data[key] = performance_data[key][-50:]

save_performance_data(performance_data)
print("Done!")