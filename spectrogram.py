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
parser.add_argument("--compare", metavar="FILE", help="Compare with another file")
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
    'lossless': {'cutoff': (21000, 48000), 'sbr': False, 'shelf': 'none'},
}

def detect_codec(data, sr):
    from scipy.signal import stft
    from scipy.ndimage import gaussian_filter1d
    
    nperseg = 8192
    noverlap = nperseg * 3 // 4
    
    frequencies, times, Zxx = stft(data, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    power = np.abs(Zxx) ** 2
    avg_spectrum = np.mean(power, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)
    
    # decimations
    time_decim = max(1, Zxx.shape[1] // 2000)
    Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decim]) ** 2 + 1e-10)
    times_decim = times[::time_decim]
    del Zxx, power
    
    smoothed = gaussian_filter1d(avg_db, sigma=3)
    gradient = np.gradient(smoothed)
    second_deriv = np.gradient(gradient)
    
    noise_floor = np.percentile(avg_db, 5)
    
    ultrasonic_energy = None
    ultrasonic_delta = None
    if sr > 48000:
        idx_24k = np.argmin(np.abs(frequencies - 24000))
        
        ultrasonic_peak = np.max(avg_db[idx_24k:])
        ultrasonic_energy = ultrasonic_peak
        ultrasonic_delta = ultrasonic_peak - noise_floor
    
    nyquist = sr / 2
    search_start_idx = np.argmin(np.abs(frequencies - 12000))
    search_end_idx = np.argmin(np.abs(frequencies - min(22000, nyquist * 0.95)))
    
    cutoff_freq = nyquist
    cutoff_idx = len(frequencies) - 1
    shelf_type = 'none'
    drop_detected = False
    
    if search_end_idx > search_start_idx:
        search_gradient = gradient[search_start_idx:search_end_idx]
        search_second = second_deriv[search_start_idx:search_end_idx]
        search_spectrum = smoothed[search_start_idx:search_end_idx]
        
        baseline_grad = np.median(gradient[search_start_idx//2:search_start_idx])
        
        threshold = baseline_grad - 1.5
        
        for i in range(len(search_gradient)):
            if search_gradient[i] < threshold and search_spectrum[i] > noise_floor + 10:
                window_start = max(0, i - 5)
                window_end = min(len(search_gradient), i + 10)
                local_drop = np.min(search_gradient[window_start:window_end])
                
                if local_drop < -1.0:
                    cutoff_idx = search_start_idx + i
                    cutoff_freq = frequencies[cutoff_idx]
                    drop_detected = True
                    
                    if local_drop < -3:
                        shelf_type = 'hard'
                    elif local_drop < -1.5:
                        shelf_type = 'medium'
                    else:
                        shelf_type = 'soft'
                    break
        
        if not drop_detected:
            end_energy = np.mean(smoothed[-20:])
            mid_energy = np.mean(smoothed[search_start_idx:search_start_idx+20])
            if end_energy > noise_floor + 5 and (mid_energy - end_energy) < 20:
                cutoff_freq = nyquist
                shelf_type = 'none'
    
    sbr_detected = False
    if cutoff_freq < 16000 and cutoff_idx < len(frequencies) - 50:
        below_start = max(0, cutoff_idx - 80)
        below_end = cutoff_idx - 20
        above_start = cutoff_idx + 20
        above_end = min(len(avg_db), cutoff_idx + 80)
        
        if below_end > below_start and above_end > above_start:
            below_region = avg_db[below_start:below_end]
            above_region = avg_db[above_start:above_end]
            
            below_energy = np.mean(below_region)
            above_energy = np.mean(above_region)
            
            if above_energy > noise_floor + 15 and (below_energy - above_energy) < 12:
                min_len = min(len(below_region), len(above_region))
                if min_len > 10:
                    corr = np.corrcoef(below_region[:min_len], above_region[:min_len])[0, 1]
                    if not np.isnan(corr) and corr > 0.5:
                        sbr_detected = True
    
    is_lossless = (cutoff_freq > nyquist * 0.9) and shelf_type == 'none'
    is_transcode = False
    transcode_warning = None
    
    if sr > 48000 and cutoff_freq < 20000 and drop_detected:
        is_transcode = True
        transcode_warning = f"High sample rate ({sr}Hz) but cutoff at {cutoff_freq:.0f}Hz - likely upsampled lossy"
    elif sr > 48000 and ultrasonic_delta is not None and ultrasonic_delta < 20:
        is_transcode = True
        transcode_warning = f"No significant ultrasonic content - likely upsampled from 44.1/48kHz source"
    elif cutoff_freq < 14000 and drop_detected and not sbr_detected:
        is_transcode = True
        transcode_warning = f"Low cutoff ({cutoff_freq:.0f}Hz) suggests heavily compressed source"
    
    scores = {}
    for codec, profile in CODEC_PROFILES.items():
        score = 0
        low, high = profile['cutoff']
        
        if low <= cutoff_freq <= high:
            score += 40
        elif abs(cutoff_freq - (low + high) / 2) < 2000:
            score += 20
        elif abs(cutoff_freq - (low + high) / 2) < 4000:
            score += 5
        
        if profile['shelf'] == shelf_type:
            score += 30
        elif profile['shelf'] in ['hard', 'medium'] and shelf_type in ['hard', 'medium']:
            score += 15
        elif profile['shelf'] in ['soft', 'none'] and shelf_type in ['soft', 'none']:
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
        'is_lossless': is_lossless,
        'is_transcode': is_transcode,
        'transcode_warning': transcode_warning,
        'ultrasonic_energy': ultrasonic_energy,
        'ultrasonic_delta': ultrasonic_delta,
        'noise_floor': noise_floor,
        'scores': scores,
        'frequencies': frequencies,
        'times': times_decim,
        'Sxx_db': Sxx_db,
        'avg_spectrum_db': avg_db,
        'nyquist': nyquist,
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
    ffprobe_info = None
    try:
        import subprocess
        result_probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path],
            capture_output=True, text=True, timeout=5
        )
        if result_probe.returncode == 0:
            import json
            ffprobe_info = json.loads(result_probe.stdout)
    except:
        pass
    
    print("[2/3] Analyzing codec signatures...")
    t0 = time.perf_counter()
    
    result = detect_codec(data, sr)
    detect_time = time.perf_counter() - t0
    
    print(f"\n{'='*50}")
    print("CODEC ANALYSIS")
    print(f"{'='*50}")
    
    if result['is_lossless']:
        print(f"  Result:            LIKELY LOSSLESS")
    elif result['is_transcode']:
        print(f"  Result:            TRANSCODED/UPSAMPLED LOSSY")
    else:
        print(f"  Closest profile:   {result['profile'].upper()}")
    
    print(f"  Confidence:        {result['confidence']} ({result['confidence_score']}/100)")
    print(f"  Cutoff frequency:  {result['cutoff_freq']:.0f} Hz")
    print(f"  Shelf type:        {result['shelf_type']}")
    print(f"  SBR detected:      {'yes' if result['sbr_detected'] else 'no'}")
    
    if result['ultrasonic_energy'] is not None:
        delta = result['ultrasonic_delta']
        if delta < 20:
            status = "EMPTY/noise only"
        elif delta < 35:
            status = "minimal"
        else:
            status = "present"
        print(f"  Ultrasonic (24k+): {delta:.1f} dB above noise ({status})")
    
    if result['transcode_warning']:
        print(f"\n  ⚠ {result['transcode_warning']}")
    
    if ffprobe_info:
        print(f"\n{'='*50}")
        print("CONTAINER METADATA (ffprobe)")
        print(f"{'='*50}")
        for stream in ffprobe_info.get('streams', []):
            if stream.get('codec_type') == 'audio':
                print(f"  Codec:        {stream.get('codec_name', 'unknown')}")
                print(f"  Sample rate:  {stream.get('sample_rate', 'unknown')} Hz")
                print(f"  Bit depth:    {stream.get('bits_per_raw_sample', stream.get('bits_per_sample', 'unknown'))}")
                print(f"  Channels:     {stream.get('channels', 'unknown')}")
                if 'bit_rate' in stream:
                    print(f"  Bitrate:      {int(stream['bit_rate'])//1000} kbps")
    
    print(f"{'='*50}")
    
    print("\nProfile scores:")
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
    nyquist = result['nyquist']
    
    ax1.plot(freqs, spectrum, 'b-', linewidth=0.8, alpha=0.7, label='Spectrum')
    ax1.axvline(x=result['cutoff_freq'], color='r', linestyle='--', label=f"Cutoff: {result['cutoff_freq']:.0f} Hz")
    if sr > 48000:
        ax1.axvline(x=24000, color='g', linestyle=':', alpha=0.5, label='24 kHz')
        ax1.axhline(y=result['noise_floor'], color='gray', linestyle=':', alpha=0.5, label='Noise floor')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    
    if result['is_lossless']:
        title = "Frequency Spectrum - LIKELY LOSSLESS"
    elif result['is_transcode']:
        title = "Frequency Spectrum - TRANSCODED LOSSY SOURCE"
    else:
        title = f"Frequency Spectrum - Closest: {result['profile'].upper()} ({result['confidence']})"
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, nyquist)
    
    extent = [result['times'][0], result['times'][-1], freqs[0], freqs[-1]]
    im = ax2.imshow(result['Sxx_db'], aspect='auto', origin='lower', extent=extent, cmap='inferno', interpolation='bilinear')
    ax2.axhline(y=result['cutoff_freq'], color='white', linestyle='--', alpha=0.7)
    if sr > 48000:
        ax2.axhline(y=24000, color='green', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')
    ax2.set_ylim(0, nyquist)
    plt.colorbar(im, ax=ax2, label='dB')
    
    plt.tight_layout()
    render_time = time.perf_counter() - t0
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(outputs_dir / f"{Path(file_path).stem}_analysis.pdf")
    
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

if args.compare:
    if not os.path.isfile(args.compare):
        print(f"Error: Comparison file not found: {args.compare}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[2/4] Loading comparison file...")
    data2, sr2 = sf.read(args.compare, dtype='float32')
    if data2.ndim > 1:
        data2 = data2.mean(axis=1)
    
    if sr2 != sr:
        from scipy.signal import resample
        data2 = resample(data2, int(len(data2) * sr / sr2))
        print(f"      Resampled {sr2} -> {sr} Hz")
    
    min_len = min(len(data), len(data2))
    data = data[:min_len]
    data2 = data2[:min_len]
    
    diff = data - data2
    diff_rms = np.sqrt(np.mean(diff**2))
    orig_rms = np.sqrt(np.mean(data**2))
    similarity = max(0, (1 - diff_rms / orig_rms)) * 100 if orig_rms > 0 else 0
    
    print(f"      Aligned to {min_len/sr:.2f}s")
    print(f"      Similarity: {similarity:.1f}%")
    print(f"      Difference RMS: {20*np.log10(diff_rms + 1e-10):.1f} dB")
    
    print("[3/4] Computing spectrograms...")
    t0 = time.perf_counter()
    
    from scipy.signal import stft
    nperseg = 2048
    noverlap = 1536
    
    frequencies, times, Z1 = stft(data, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Z2 = stft(data2, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, Zd = stft(diff, fs=sr, nperseg=nperseg, noverlap=noverlap)
    
    time_decim = max(1, Z1.shape[1] // 2000)
    S1_db = 10 * np.log10(np.abs(Z1[:, ::time_decim])**2 + 1e-10)
    S2_db = 10 * np.log10(np.abs(Z2[:, ::time_decim])**2 + 1e-10)
    Sdiff_db = 10 * np.log10(np.abs(Zd[:, ::time_decim])**2 + 1e-10)
    times = times[::time_decim]
    
    del Z1, Z2, Zd, data, data2, diff
    
    stft_time = time.perf_counter() - t0
    print(f"      Done ({fmt_time(stft_time)})")
    
    print("[4/4] Rendering...")
    t0 = time.perf_counter()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    extent = [times[0], times[-1], frequencies[0], frequencies[-1]]
    vmin, vmax = -80, 0
    
    im1 = ax1.imshow(S1_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax)
    ax1.set_ylabel('Freq (Hz)')
    ax1.set_title(f'A: {Path(file_path).stem}')
    ax1.set_ylim(0, min(20000, sr/2))
    plt.colorbar(im1, ax=ax1, label='dB')
    
    im2 = ax2.imshow(S2_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Freq (Hz)')
    ax2.set_title(f'B: {Path(args.compare).stem}')
    ax2.set_ylim(0, min(20000, sr/2))
    plt.colorbar(im2, ax=ax2, label='dB')
    
    im3 = ax3.imshow(Sdiff_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', vmin=-100, vmax=-20)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Freq (Hz)')
    ax3.set_title(f'Difference (A - B) | Similarity: {similarity:.1f}%')
    ax3.set_ylim(0, min(20000, sr/2))
    plt.colorbar(im3, ax=ax3, label='dB')
    
    plt.tight_layout()
    render_time = time.perf_counter() - t0
    
    if args.output:
        output_path = args.output
    else:
        output_path = str(outputs_dir / f"{Path(file_path).stem}_vs_{Path(args.compare).stem}.pdf")
    
    plt.savefig(output_path, dpi=150)
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

target_time_bins = 4000
hop = NPERSEG - noverlap
total_frames = len(data) // hop
time_decimation = max(1, total_frames // target_time_bins)

frequencies, times, Zxx = stft(data, fs=sr, nperseg=NPERSEG, noverlap=noverlap, window='hann')

# conversion then ERADICATE!!!! to save memory lol
Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decimation]) ** 2 + 1e-10)
times = times[::time_decimation]
del Zxx, data

stft_time = time.perf_counter() - t0

print(f"      {Sxx_db.shape[0]}x{Sxx_db.shape[1]} bins, {sr/NPERSEG:.1f} Hz res ({fmt_time(stft_time)})")

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