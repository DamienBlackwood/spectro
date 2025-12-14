#!/usr/bin/env python3
import time
script_start = time.perf_counter()

import numpy as np
import soundfile as sf
import argparse
import sys
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Spectrogram Generator")
parser.add_argument("file_path", nargs="?", help="Input audio file")
parser.add_argument("-o", "--output", help="Output filename")
parser.add_argument("--detect", action="store_true", help="Codec detection mode")
parser.add_argument("--quality", action="store_true", help="Full quality mode (slower)")
parser.add_argument("--log", action="store_true", help="Log frequency axis")
parser.add_argument("--no-display", action="store_true", help="Don't show plot")
parser.add_argument("--info", action="store_true", help="Show file info only")
args = parser.parse_args()

if args.quality:
    MODE = "quality"
    MAX_SR = None
    NPERSEG = 2048
    OVERLAP = 0.75
    DPI = 300
    FMT = "png"
    MAX_TIME_BINS = None
    MAX_FREQ_BINS = None
else:
    MODE = "fast"
    MAX_SR = 44100
    NPERSEG = 1024
    OVERLAP = 0.5
    DPI = 150
    FMT = "pdf"
    MAX_TIME_BINS = 2000
    MAX_FREQ_BINS = 512

CODEC_PROFILES = {
    'mp3_128': {'cutoff': (15500, 16500), 'sbr': False, 'shelf': 'hard'},
    'mp3_192': {'cutoff': (18000, 19000), 'sbr': False, 'shelf': 'hard'},
    'mp3_256': {'cutoff': (19500, 20500), 'sbr': False, 'shelf': 'hard'},
    'mp3_320': {'cutoff': (20000, 21000), 'sbr': False, 'shelf': 'medium'},
    'aac_128': {'cutoff': (15000, 16500), 'sbr': False, 'shelf': 'soft'},
    'aac_256': {'cutoff': (19000, 20500), 'sbr': False, 'shelf': 'soft'},
    'he_aac': {'cutoff': (13000, 15000), 'sbr': True, 'shelf': 'soft'},
    'opus_128': {'cutoff': (19000, 20500), 'sbr': False, 'shelf': 'soft'},
    'vorbis_128': {'cutoff': (15500, 17000), 'sbr': False, 'shelf': 'medium'},
}

def detect_codec(data, sr):
    from scipy.signal import stft
    
    frequencies, times, Zxx = stft(data, fs=sr, nperseg=8192, noverlap=6144, window='hann')
    power = np.abs(Zxx) ** 2
    avg_spectrum = np.mean(power, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)
    
    noise_floor = np.percentile(avg_db, 10)
    peak_level = np.max(avg_db)
    threshold = noise_floor + (peak_level - noise_floor) * 0.1
    
    cutoff_freq = None
    for i in range(len(frequencies) - 1, 0, -1):
        if avg_db[i] > threshold:
            cutoff_freq = frequencies[i]
            break
    
    if cutoff_freq is None:
        cutoff_freq = frequencies[-1]
    
    cutoff_idx = np.argmin(np.abs(frequencies - cutoff_freq))
    
    if cutoff_idx > 100:
        far_below = avg_db[cutoff_idx-100:cutoff_idx-50]
        near_cutoff = avg_db[cutoff_idx-20:cutoff_idx]
        at_cutoff = avg_db[cutoff_idx:cutoff_idx+20] if cutoff_idx+20 < len(avg_db) else avg_db[cutoff_idx:]
        
        if len(far_below) > 0 and len(near_cutoff) > 0 and len(at_cutoff) > 0:
            gradual_drop = np.mean(far_below) - np.mean(near_cutoff)
            sudden_drop = np.mean(near_cutoff) - np.mean(at_cutoff)
            
            if sudden_drop > 15:
                shelf_type = 'hard'
            elif sudden_drop > 8:
                shelf_type = 'medium'
            elif sudden_drop > 3 or gradual_drop > 10:
                shelf_type = 'soft'
            else:
                shelf_type = 'none'
        else:
            shelf_type = 'none'
    else:
        shelf_type = 'none'
    
    sbr_detected = False
    if cutoff_freq < 16000 and cutoff_idx < len(frequencies) - 50:
        below_region = avg_db[max(0, cutoff_idx-80):cutoff_idx-20]
        above_region = avg_db[cutoff_idx+20:min(len(avg_db), cutoff_idx+80)]
        
        if len(above_region) > 20 and len(below_region) > 20:
            below_energy = np.mean(below_region)
            above_energy = np.mean(above_region)
            energy_diff = below_energy - above_energy
            
            if energy_diff < 15 and above_energy > noise_floor + 10:
                min_len = min(len(below_region), len(above_region))
                below_norm = below_region[:min_len] - np.mean(below_region[:min_len])
                above_norm = above_region[:min_len] - np.mean(above_region[:min_len])
                
                if np.std(below_norm) > 0 and np.std(above_norm) > 0:
                    correlation = np.corrcoef(below_norm, above_norm)[0, 1]
                    if correlation > 0.6:
                        sbr_detected = True
    
    scores = {}
    for codec, profile in CODEC_PROFILES.items():
        score = 0
        
        low, high = profile['cutoff']
        if low <= cutoff_freq <= high:
            score += 40
        elif low - 1000 <= cutoff_freq <= high + 1000:
            score += 20
        elif low - 2000 <= cutoff_freq <= high + 2000:
            score += 5
        
        if profile['shelf'] == shelf_type:
            score += 30
        elif (profile['shelf'] in ['hard', 'medium'] and shelf_type in ['hard', 'medium']):
            score += 15
        elif (profile['shelf'] in ['soft', 'none'] and shelf_type in ['soft', 'none']):
            score += 15
        
        if profile['sbr'] == sbr_detected:
            score += 30
        
        scores[codec] = score
    
    best_profile = max(scores, key=scores.get)
    confidence = scores[best_profile]
    
    if confidence < 40:
        confidence_str = "low"
    elif confidence < 70:
        confidence_str = "medium"
    else:
        confidence_str = "high"
    
    return {
        'profile': best_profile,
        'confidence': confidence_str,
        'confidence_score': confidence,
        'cutoff_freq': cutoff_freq,
        'shelf_type': shelf_type,
        'sbr_detected': sbr_detected,
        'scores': scores,
        'frequencies': frequencies,
        'avg_spectrum_db': avg_db,
    }

def fmt_time(s):
    if s < 60: return f"{s:.2f}s"
    return f"{int(s//60)}m {s%60:.1f}s"

def decimate_2d(arr, max_rows, max_cols):
    rows, cols = arr.shape
    rf = max(1, rows // max_rows) if max_rows else 1
    cf = max(1, cols // max_cols) if max_cols else 1
    if rf == 1 and cf == 1:
        return arr, 1, 1
    return arr[::rf, ::cf], rf, cf

if args.file_path is None:
    args.file_path = input("Audio file: ").strip()

file_path = args.file_path.strip().strip('"\'')
if not os.path.isfile(file_path):
    print(f"Error: File not found: {file_path}", file=sys.stderr)
    sys.exit(1)

outputs_dir = Path("outputs/spectrograms")
outputs_dir.mkdir(parents=True, exist_ok=True)

file_size = Path(file_path).stat().st_size
print(f"\n[{MODE.upper()} MODE] {Path(file_path).name} ({file_size/1024/1024:.1f} MB)")

print("[1/4] Loading...")
t0 = time.perf_counter()
data, sr = sf.read(file_path, dtype='float32')
load_time = time.perf_counter() - t0

if data.ndim > 1:
    data = data.mean(axis=1)

duration = len(data) / sr
print(f"      {sr} Hz, {duration:.1f}s, {len(data):,} samples ({fmt_time(load_time)})")

if args.info:
    sys.exit(0)

if args.detect:
    print("[2/3] Analyzing codec signatures...")
    t0 = time.perf_counter()
    
    result = detect_codec(data, sr)
    detect_time = time.perf_counter() - t0
    
    print(f"\n{'='*50}")
    print("LOSSY SOURCE ANALYSIS (HEURISTIC)")
    print(f"{'='*50}")
    print(f"  Closest profile:   {result['profile'].upper()}")
    print(f"  Confidence level:  {result['confidence']} ({result['confidence_score']}/100)")
    print(f"  Cutoff frequency:  {result['cutoff_freq']:.0f} Hz")
    print(f"  Shelf type:        {result['shelf_type']}")
    print(f"  SBR detected:      {'yes' if result['sbr_detected'] else 'no'}")
    print(f"{'='*50}")
    
    print("\nClosest matching profiles:")
    sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    for codec, score in sorted_scores:
        print(f"  {codec:15} {score:3}/100")
    
    print(f"\n[3/3] Generating analysis plot...")
    t0 = time.perf_counter()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    freqs = result['frequencies']
    spectrum = result['avg_spectrum_db']
    
    ax1.plot(freqs, spectrum, 'b-', linewidth=0.8)
    ax1.axvline(x=result['cutoff_freq'], color='r', linestyle='--', label=f"Cutoff: {result['cutoff_freq']:.0f} Hz")
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title(
        f"Frequency Spectrum - Closest match: {result['profile'].upper()} "
        f"({result['confidence']} confidence)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(24000, sr/2))
    
    from scipy.signal import stft
    frequencies, times, Zxx = stft(data, fs=sr, nperseg=2048, noverlap=1536, window='hann')
    Sxx_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
    
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    im = ax2.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', interpolation='bilinear')
    ax2.axhline(y=result['cutoff_freq'], color='white', linestyle='--', alpha=0.7, label=f"Cutoff: {result['cutoff_freq']:.0f} Hz")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram with inferred cutoff (heuristic)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, min(24000, sr/2))
    plt.colorbar(im, ax=ax2, label='dB')
    
    plt.tight_layout()
    render_time = time.perf_counter() - t0
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(outputs_dir / f"{Path(file_path).stem}_codec_analysis.pdf")
    
    t0 = time.perf_counter()
    plt.savefig(output_path, dpi=150)
    save_time = time.perf_counter() - t0
    
    print(f"      Saved: {output_path}")
    
    if not args.no_display:
        try: plt.show()
        except: pass
    
    plt.close('all')
    
    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
    sys.exit(0)

print("[2/4] Preprocessing...")
t0 = time.perf_counter()
original_sr = sr

if MAX_SR and sr > MAX_SR:
    if sr % MAX_SR == 0:
        factor = sr // MAX_SR
        data = data[::factor]
        sr = MAX_SR
        print(f"      Decimated {original_sr} → {sr} Hz")
    else:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, MAX_SR)
        data = resample_poly(data, MAX_SR // g, sr // g)
        sr = MAX_SR
        print(f"      Resampled {original_sr} → {sr} Hz")

preprocess_time = time.perf_counter() - t0

print("[3/4] Computing STFT...")
t0 = time.perf_counter()

from scipy.signal import stft
noverlap = int(NPERSEG * OVERLAP)
frequencies, times, Zxx = stft(data, fs=sr, nperseg=NPERSEG, noverlap=noverlap, window='hann')

Sxx_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
stft_time = time.perf_counter() - t0

print(f"      {Sxx_db.shape[0]}x{Sxx_db.shape[1]} bins, {sr/NPERSEG:.1f} Hz res ({fmt_time(stft_time)})")

if MAX_TIME_BINS or MAX_FREQ_BINS:
    orig_shape = Sxx_db.shape
    Sxx_db, rf, cf = decimate_2d(Sxx_db, MAX_FREQ_BINS, MAX_TIME_BINS)
    if rf > 1 or cf > 1:
        frequencies = frequencies[::rf]
        times = times[::cf]
        print(f"      Decimated to {Sxx_db.shape[0]}x{Sxx_db.shape[1]} for display")

print("[4/4] Rendering...")
t0 = time.perf_counter()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
im = ax.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', interpolation='bilinear')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'{Path(file_path).stem}')
plt.colorbar(im, ax=ax, label='dB')

if args.log:
    ax.set_yscale('log')
    ax.set_ylim(20, frequencies[-1])

plt.tight_layout()
render_time = time.perf_counter() - t0

if args.output:
    output_path = args.output
else:
    output_path = str(outputs_dir / f"{Path(file_path).stem}.{FMT}")

t0 = time.perf_counter()
plt.savefig(output_path, dpi=DPI, format=FMT)
save_time = time.perf_counter() - t0

print(f"      Saved: {output_path} ({fmt_time(save_time)})")

if not args.no_display:
    try: plt.show()
    except: pass

plt.close('all')

total = time.perf_counter() - script_start
print(f"\nDone in {fmt_time(total)} (load:{fmt_time(load_time)} stft:{fmt_time(stft_time)} render:{fmt_time(render_time)} save:{fmt_time(save_time)})")