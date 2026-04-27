#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import argparse
import time

# Quick audio check before heavy imports
_parser = argparse.ArgumentParser(description="Spectrogram Generator")
_parser.add_argument("file_path", nargs="?", help="Input audio file")
_parser.add_argument("-o", "--output", help="Output filename")
_parser.add_argument("--detect", action="store_true", help="Spectral authenticity / transcode-evidence analysis")
_parser.add_argument("--compare", metavar="FILE", help="Compare with another audio file")
_parser.add_argument("--log", action="store_true", help="Log frequency axis")
_parser.add_argument("--no-open", dest="no_open", action="store_true", help="Don't auto-open output file")
_parser.add_argument("--info", action="store_true", help="Show file info only")
_parser.add_argument("--json", nargs="?", const=None, metavar="FILE", help="Write machine-readable JSON report")
_args, _ = _parser.parse_known_args()
if _args.file_path:
    _fp = _args.file_path.strip().strip('"\'')
    if os.path.isfile(_fp):
        with open(_fp, 'rb') as _f:
            _head = _f.read(16)
        _ext = Path(_fp).suffix.lower()
        _audio_magic = {b'RIFF', b'fLaC', b'OggS', b'MThd'}
        _audio_exts = {'.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aac', '.wma', '.aiff', '.aif', '.ape', '.wv', '.mpc', '.dff', '.dsf', '.caf'}
        if not (any(_head.startswith(m) for m in _audio_magic) or _ext in _audio_exts):
            print(f"Error: '{Path(_fp).name}' does not appear to be an audio file", file=sys.stderr)
            sys.exit(1)
del _parser, _args, _fp, _head, _ext, _audio_magic, _audio_exts

import platform

_import_start = time.perf_counter()
import numpy as np
import soundfile as sf
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft, correlate, resample
from scipy.ndimage import gaussian_filter1d

_import_time = time.perf_counter() - _import_start

parser = argparse.ArgumentParser(description="Spectrogram Generator")
parser.add_argument("file_path", nargs="?", help="Input audio file")
parser.add_argument("-o", "--output", help="Output filename")
parser.add_argument("--detect", action="store_true", help="Spectral authenticity / transcode-evidence analysis")
parser.add_argument("--compare", metavar="FILE", help="Compare with another audio file")
parser.add_argument("--log", action="store_true", help="Log frequency axis")
parser.add_argument("--no-open", dest="no_open", action="store_true", help="Don't auto-open output file")
parser.add_argument("--info", action="store_true", help="Show file info only")
parser.add_argument("--json", nargs="?", const=None, metavar="FILE", help="Write machine-readable JSON report")

CODEC_PROFILES = {
    'mp3_128':   {'cutoff': (15500, 16500), 'sbr': False, 'shelf': 'hard'},
    'mp3_192':   {'cutoff': (18000, 19000), 'sbr': False, 'shelf': 'hard'},
    'mp3_256':   {'cutoff': (19500, 20500), 'sbr': False, 'shelf': 'hard'},
    'mp3_320':   {'cutoff': (20000, 21000), 'sbr': False, 'shelf': 'medium'},
    'aac_128':   {'cutoff': (15000, 16500), 'sbr': False, 'shelf': 'soft'},
    'aac_256':   {'cutoff': (19000, 20500), 'sbr': False, 'shelf': 'soft'},
    'he_aac':    {'cutoff': (13000, 15000), 'sbr': True,  'shelf': 'soft'},
    'opus_128':  {'cutoff': (19000, 20500), 'sbr': False, 'shelf': 'soft'},
    'vorbis_128':{'cutoff': (15500, 17000), 'sbr': False, 'shelf': 'medium'},
}

DR_THRESHOLDS = {
    'brickwalled': 6,
    'compressed': 9,
    'moderate': 14
}

@dataclass
class DynamicsResult:
    peak_db: float
    rms_db: float
    crest_factor: float
    dynamic_range: float
    dr_rating: str
    clipped_samples: int
    clip_percentage: float
    clip_times: List[float]

@dataclass
class EvidenceFlag:
    severity: str  # info, low, medium, high, critical
    name: str
    detail: str

@dataclass
class TranscodeEvidence:
    verdict: str
    explanation: str
    cutoff_freq: float
    cutoff_persistence: float
    hard_cutoff: bool
    high_band_db: float
    near_nyquist_db: float
    noise_floor: float
    suspicious_flags: List[str]
    edge_p10: float
    edge_p50: float
    edge_p90: float
    best_drop_freq: float
    max_drop_db: float
    sbr_likelihood: str
    active_frames_pct: float
    suspicious_windows: List[str]

@dataclass
class SpectralAnalysisResult:
    profile: str
    confidence: str
    confidence_score: int
    cutoff_freq: float
    shelf_type: str
    sbr_likelihood: str
    transcode_suspected: bool
    transcode_warning: Optional[str]
    ultrasonic_energy: Optional[float]
    ultrasonic_delta: Optional[float]
    noise_floor: float
    scores: Dict[str, int]
    frequencies: np.ndarray
    times: np.ndarray
    Sxx_db: np.ndarray
    avg_spectrum_db: np.ndarray
    nyquist: float
    evidence: TranscodeEvidence
    edge_times: np.ndarray
    edge_values: np.ndarray

def analyze_dynamics(data: np.ndarray, sr: int) -> DynamicsResult:
    """Analyze dynamic range, clipping, and crest factor."""
    peak = np.max(np.abs(data))
    peak_db = 20 * np.log10(peak + 1e-10)
    
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    
    crest_factor = peak_db - rms_db
    
    clip_threshold = 0.99
    clipped_samples = np.sum(np.abs(data) >= clip_threshold)
    clip_percentage = (clipped_samples / data.size) * 100
    
    # Mix to mono for macro-level windowed dynamics
    mono_data = data.mean(axis=1) if data.ndim > 1 else data
    
    clip_indices = np.where(np.abs(mono_data) >= clip_threshold)[0]
    clip_times = (clip_indices / sr).tolist() if len(clip_indices) > 0 else []
    
    window_size = int(0.05 * sr)
    n_windows = len(mono_data) // window_size
    if n_windows > 0:
        windowed = mono_data[:n_windows * window_size].reshape(n_windows, window_size)
        window_rms = np.sqrt(np.mean(windowed ** 2, axis=1))
        window_rms_db = 20 * np.log10(window_rms + 1e-10)
        
        loud = np.percentile(window_rms_db, 95)
        quiet = np.percentile(window_rms_db, 5)
        dynamic_range = loud - quiet
    else:
        dynamic_range = 0
    
    # "Loudness war era" recognition
    dr_rating = "dynamic"
    if crest_factor < DR_THRESHOLDS['brickwalled']:
        dr_rating = "brickwalled"
    elif crest_factor < DR_THRESHOLDS['compressed']:
        dr_rating = "compressed"
    elif crest_factor < DR_THRESHOLDS['moderate']:
        dr_rating = "moderate"
    
    return DynamicsResult(
        peak_db=peak_db, rms_db=rms_db, crest_factor=crest_factor,
        dynamic_range=dynamic_range, dr_rating=dr_rating,
        clipped_samples=clipped_samples, clip_percentage=clip_percentage,
        clip_times=clip_times[:10] if len(clip_times) > 0 else []
    )

def open_file(path: str) -> None:
    """Open file with system default application."""
    try:
        if platform.system() == 'Darwin':
            subprocess.run(['open', path], check=True)
        elif platform.system() == 'Windows':
            os.startfile(path)
        else:
            subprocess.run(['xdg-open', path], check=True)
    except Exception:
        pass

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio with soundfile, fallback to FFmpeg for unsupported formats."""
    try:
        return sf.read(file_path, dtype='float32')
    except (sf.LibsndfileError, OSError, RuntimeError):
        import shutil, tempfile
        if not shutil.which('ffmpeg'):
            print("Error: FFmpeg not found. Install it to handle this file format.", file=sys.stderr)
            sys.exit(1)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-v', 'error', '-i', file_path,
                 '-acodec', 'pcm_f32le', tmp_path],
                check=True, capture_output=True
            )
            return sf.read(tmp_path, dtype='float32')
        except subprocess.CalledProcessError:
            print(f"Error: Could not decode '{file_path}' — not a valid audio file.", file=sys.stderr)
            sys.exit(1)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

def active_band_edge(S_db: np.ndarray, freqs: np.ndarray, floor_margin_db: float = 18) -> np.ndarray:
    """Estimate highest frequency with meaningful energy per frame."""
    edges = []
    for frame in S_db.T:
        noise_floor = np.percentile(frame, 10)
        active = frame > noise_floor + floor_margin_db
        if not np.any(active):
            edges.append(np.nan)
            continue
        edges.append(freqs[np.where(active)[0][-1]])
    return np.array(edges)

def band_mean(avg_db: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.nan
    return float(np.mean(avg_db[mask]))

def cutoff_drop_score(avg_db: np.ndarray, freqs: np.ndarray, cutoff_hz: float, width: float = 1000) -> float:
    below = band_mean(avg_db, freqs, cutoff_hz - width, cutoff_hz)
    above = band_mean(avg_db, freqs, cutoff_hz, cutoff_hz + width)
    if np.isnan(below) or np.isnan(above):
        return 0.0
    return below - above

def get_active_frames(Sxx_db: np.ndarray, freqs: np.ndarray, noise_floor: float, nyquist: float) -> np.ndarray:
    """Return boolean mask of frames that are loud enough and spectrally rich."""
    frame_mean = np.mean(Sxx_db, axis=0)
    high_mask = (freqs >= 8000) & (freqs < min(16000, nyquist * 0.90))
    if not np.any(high_mask):
        high_band_energy = np.full(Sxx_db.shape[1], noise_floor - 100)
    else:
        high_band_energy = np.mean(Sxx_db[high_mask, :], axis=0)
    active = (frame_mean > noise_floor + 12) & (high_band_energy > noise_floor + 6)
    return active

def _analyze_channel(data_ch: np.ndarray, sr: int) -> Tuple[float, str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analyze a single channel and return cutoff, shelf, SBR likelihood, and spectral data."""
    nperseg = 8192
    noverlap = 6144
    
    frequencies, times, Zxx = stft(data_ch, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
    power = np.abs(Zxx) ** 2
    avg_spectrum = np.mean(power, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)
    
    time_decim = max(1, Zxx.shape[1] // 2000)
    Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decim]) ** 2 + 1e-10)
    times_decim = times[::time_decim]
    
    smoothed = gaussian_filter1d(avg_db, sigma=3)
    gradient = np.gradient(smoothed)
    noise_floor = np.percentile(avg_db, 5)
    
    nyquist = sr / 2
    search_start_idx = np.argmin(np.abs(frequencies - 12000))
    search_end_hz = nyquist * 0.995
    search_end_idx = np.searchsorted(frequencies, search_end_hz)
    if search_end_idx <= search_start_idx:
        search_end_idx = len(frequencies) - 1
    
    cutoff_freq = nyquist
    cutoff_idx = len(frequencies) - 1
    shelf_type = 'none'
    drop_detected = False
    
    if search_end_idx > search_start_idx:
        search_gradient = gradient[search_start_idx:search_end_idx]
        search_spectrum = smoothed[search_start_idx:search_end_idx]
        baseline_grad = np.median(gradient[search_start_idx // 2:search_start_idx])
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
            mid_energy = np.mean(smoothed[search_start_idx:search_start_idx + 20])
            if end_energy > noise_floor + 5 and (mid_energy - end_energy) < 20:
                cutoff_freq = nyquist
                shelf_type = 'none'
    
    sbr_likelihood = "none"
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
                        sbr_likelihood = "possible"
                        upper_flatness = spectral_flatness(10 ** (above_region / 10))
                        lower_flatness = spectral_flatness(10 ** (below_region / 10))
                        if upper_flatness > lower_flatness * 1.3:
                            sbr_likelihood = "likely"
    
    return cutoff_freq, shelf_type, sbr_likelihood, frequencies, times_decim, Sxx_db, avg_db, smoothed

def spectral_flatness(power_band: np.ndarray) -> float:
    geometric = np.exp(np.mean(np.log(power_band + 1e-12)))
    arithmetic = np.mean(power_band + 1e-12)
    return geometric / arithmetic

def analyze_time_windows(data_ch: np.ndarray, sr: int, frequencies: np.ndarray,
                         nyquist: float, noise_floor: float) -> List[str]:
    """Analyze ~5-second windows for suspicious cutoffs."""
    window_samples = int(5 * sr)
    n_windows = max(1, len(data_ch) // window_samples)
    suspicious = []
    
    for i in range(n_windows):
        start = i * window_samples
        end = min(start + window_samples, len(data_ch))
        if end - start < window_samples // 2:
            continue
        
        segment = data_ch[start:end]
        nperseg = 8192
        noverlap = 6144
        _, _, Zxx = stft(segment, fs=sr, nperseg=nperseg, noverlap=noverlap, window='hann')
        power = np.abs(Zxx) ** 2
        avg_spectrum = np.mean(power, axis=1)
        avg_db = 10 * np.log10(avg_spectrum + 1e-10)
        smoothed = gaussian_filter1d(avg_db, sigma=3)
        gradient = np.gradient(smoothed)
        
        search_start_idx = np.argmin(np.abs(frequencies - 12000))
        search_end_idx = np.searchsorted(frequencies, nyquist * 0.995)
        if search_end_idx <= search_start_idx:
            continue
        
        search_gradient = gradient[search_start_idx:search_end_idx]
        search_spectrum = smoothed[search_start_idx:search_end_idx]
        baseline_grad = np.median(gradient[search_start_idx // 2:search_start_idx])
        threshold = baseline_grad - 1.5
        
        for j in range(len(search_gradient)):
            if search_gradient[j] < threshold and search_spectrum[j] > noise_floor + 10:
                window_start = max(0, j - 5)
                window_end = min(len(search_gradient), j + 10)
                local_drop = np.min(search_gradient[window_start:window_end])
                if local_drop < -3:
                    cf = frequencies[search_start_idx + j]
                    if cf < nyquist * 0.92:
                        t0 = start / sr
                        t1 = end / sr
                        suspicious.append(f"{t0:.0f}s–{t1:.0f}s  cutoff around {cf:.0f} Hz")
                    break
    
    return suspicious

def classify_transcode(
    cutoff_freq: float,
    shelf_type: str,
    hard_cutoff: bool,
    cutoff_persistence: float,
    high_band_db: float,
    near_nyquist_db: float,
    noise_floor: float,
    nyquist: float,
    sr: int,
    ultrasonic_delta: Optional[float],
    sbr_likelihood: str
) -> Tuple[TranscodeEvidence, List[EvidenceFlag]]:
    flags: List[EvidenceFlag] = []
    
    if sr > 48000:
        if ultrasonic_delta is not None and ultrasonic_delta < 20:
            flags.append(EvidenceFlag(
                severity="info",
                name="Limited ultrasonic content",
                detail=f"{sr} Hz file has little content above 24 kHz; may indicate 44.1/48 kHz source, not necessarily lossy."
            ))
    
    if sbr_likelihood == "likely":
        flags.append(EvidenceFlag(
            severity="medium",
            name="SBR-like reconstruction",
            detail="Upper band shows noise-like correlation with lower band, consistent with HE-AAC-style SBR."
        ))
    elif sbr_likelihood == "possible":
        flags.append(EvidenceFlag(
            severity="low",
            name="Possible SBR-like reconstruction",
            detail="Weak correlation between lower and upper band; could be SBR or natural spectral structure."
        ))
    
    if hard_cutoff and cutoff_freq < 18000 and cutoff_persistence > 0.5:
        flags.append(EvidenceFlag(
            severity="critical",
            name="Persistent hard cutoff below 18 kHz",
            detail=f"Hard cutoff around {cutoff_freq:.0f} Hz persists in {cutoff_persistence*100:.0f}% of active frames."
        ))
        verdict = "FAIL"
    elif hard_cutoff and cutoff_freq < nyquist * 0.92 and cutoff_persistence > 0.4:
        flags.append(EvidenceFlag(
            severity="high",
            name="Persistent hard cutoff below full-band range",
            detail=f"Hard cutoff around {cutoff_freq:.0f} Hz persists in {cutoff_persistence*100:.0f}% of active frames."
        ))
        verdict = "WARN"
    elif near_nyquist_db > noise_floor + 8 and not hard_cutoff:
        verdict = "PASS"
    elif high_band_db < noise_floor + 6:
        flags.append(EvidenceFlag(
            severity="low",
            name="Little high-frequency content",
            detail="Could be source/mastering choice or lossy encode."
        ))
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    
    if verdict == "PASS":
        explanation = "no obvious lossy transcode signature"
    elif verdict == "WARN":
        explanation = "suspicious cutoff / possible lossy source"
    elif verdict == "FAIL":
        explanation = "strong lossy transcode evidence"
    else:
        explanation = "not enough high-frequency information to judge"
    
    evidence = TranscodeEvidence(
        verdict=verdict,
        explanation=explanation,
        cutoff_freq=cutoff_freq,
        cutoff_persistence=cutoff_persistence,
        hard_cutoff=hard_cutoff,
        high_band_db=high_band_db,
        near_nyquist_db=near_nyquist_db,
        noise_floor=noise_floor,
        suspicious_flags=[],
        edge_p10=0.0,
        edge_p50=0.0,
        edge_p90=0.0,
        best_drop_freq=0.0,
        max_drop_db=0.0,
        sbr_likelihood=sbr_likelihood,
        active_frames_pct=0.0,
        suspicious_windows=[],
    )
    return evidence, flags

def analyze_transcode_evidence(data: np.ndarray, sr: int) -> SpectralAnalysisResult:
    if data.ndim == 1:
        channels = [data]
    else:
        channels = [data[:, ch] for ch in range(data.shape[1])]
    
    channel_results = []
    for ch in channels:
        channel_results.append(_analyze_channel(ch, sr))
    
    def suspicion_key(res):
        cf, st, *_ = res
        shelf_rank = {'hard': 0, 'medium': 1, 'soft': 2, 'none': 3}
        return (shelf_rank.get(st, 3), cf)
    
    channel_results.sort(key=suspicion_key)
    primary = channel_results[0]
    cutoff_freq, shelf_type, sbr_likelihood, frequencies, times_decim, Sxx_db, avg_db, smoothed = primary
    
    all_avg_db = np.array([r[6] for r in channel_results])
    avg_db = np.mean(all_avg_db, axis=0)
    smoothed = gaussian_filter1d(avg_db, sigma=3)
    
    noise_floor = np.percentile(avg_db, 5)
    nyquist = sr / 2
    
    _, _, _, _, _, Sxx_db_primary, _, _ = channel_results[0]
    edges_all = active_band_edge(Sxx_db_primary, frequencies, floor_margin_db=18)
    
    active_mask = get_active_frames(Sxx_db_primary, frequencies, noise_floor, nyquist)
    active_frames_pct = float(np.mean(active_mask)) if len(active_mask) > 0 else 0.0
    valid_edges = edges_all[~np.isnan(edges_all) & active_mask] if np.any(active_mask) else edges_all[~np.isnan(edges_all)]
    
    edge_p10 = float(np.nanpercentile(edges_all, 10))
    edge_p50 = float(np.nanpercentile(edges_all, 50))
    edge_p90 = float(np.nanpercentile(edges_all, 90))
    
    high_band_db = band_mean(avg_db, frequencies, 16000, min(20000, nyquist * 0.90))
    near_nyquist_db = band_mean(avg_db, frequencies, nyquist * 0.90, nyquist * 0.99)
    if np.isnan(high_band_db):
        high_band_db = noise_floor
    if np.isnan(near_nyquist_db):
        near_nyquist_db = noise_floor
    
    candidate_cutoffs = [14000, 16000, 18000, 19000, 20000, 20500, 21000]
    drops = {c: cutoff_drop_score(avg_db, frequencies, c) for c in candidate_cutoffs}
    best_drop_freq, max_drop = max(drops.items(), key=lambda x: x[1]) if drops else (0.0, 0.0)
    hard_cutoff = max_drop > 15.0
    
    if hard_cutoff:
        persistence_limit = best_drop_freq + 500
    else:
        persistence_limit = nyquist * 0.94
    cutoff_persistence = float(np.mean(valid_edges < persistence_limit)) if len(valid_edges) > 0 else 0.0
    
    ultrasonic_energy = None
    ultrasonic_delta = None
    if sr > 48000:
        idx_24k = np.searchsorted(frequencies, 24000)
        if idx_24k < len(frequencies):
            ultrasonic_peak = float(np.max(avg_db[idx_24k:]))
            ultrasonic_energy = ultrasonic_peak
            ultrasonic_delta = ultrasonic_peak - noise_floor
    
    suspicious_windows = analyze_time_windows(channels[0], sr, frequencies, nyquist, noise_floor)
    
    evidence, flags = classify_transcode(
        cutoff_freq=cutoff_freq,
        shelf_type=shelf_type,
        hard_cutoff=hard_cutoff,
        cutoff_persistence=cutoff_persistence,
        high_band_db=high_band_db,
        near_nyquist_db=near_nyquist_db,
        noise_floor=noise_floor,
        nyquist=nyquist,
        sr=sr,
        ultrasonic_delta=ultrasonic_delta,
        sbr_likelihood=sbr_likelihood
    )
    evidence.edge_p10 = edge_p10
    evidence.edge_p50 = edge_p50
    evidence.edge_p90 = edge_p90
    evidence.best_drop_freq = best_drop_freq
    evidence.max_drop_db = max_drop
    evidence.active_frames_pct = active_frames_pct
    evidence.suspicious_windows = suspicious_windows
    
    # Profile scoring for "resemblance" only
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
        
        if profile['sbr'] != (sbr_likelihood != "none"):
            score -= 40
        else:
            score += 30
            
        scores[codec] = score
    
    best_profile = max(scores, key=scores.get)
    confidence_val = scores[best_profile]
    
    if confidence_val < 40:
        confidence_str = "low"
    elif confidence_val < 70:
        confidence_str = "medium"
    else:
        confidence_str = "high"
    
    # Determine if transcode is suspected
    transcode_suspected = False
    transcode_warning = None
    
    if sr > 48000 and cutoff_freq < 20000 and hard_cutoff:
        transcode_suspected = True
        transcode_warning = f"High sample rate ({sr} Hz) but cutoff at {cutoff_freq:.0f} Hz — likely upsampled lossy"
    elif sr > 48000 and ultrasonic_delta is not None and ultrasonic_delta < 20:
        transcode_warning = (
            "High sample-rate file has little ultrasonic content — suggests a 44.1/48 kHz source "
            "or upsampled delivery, not necessarily lossy compression"
        )
    elif cutoff_freq < 14000 and hard_cutoff and sbr_likelihood == "none":
        transcode_suspected = True
        transcode_warning = f"Low cutoff ({cutoff_freq:.0f} Hz) suggests heavily compressed source"
    
    for f in flags:
        evidence.suspicious_flags.append(f"[{f.severity.upper()}] {f.name}: {f.detail}")
    
    return SpectralAnalysisResult(
        profile=best_profile, confidence=confidence_str, confidence_score=confidence_val,
        cutoff_freq=cutoff_freq, shelf_type=shelf_type, sbr_likelihood=sbr_likelihood,
        transcode_suspected=transcode_suspected, transcode_warning=transcode_warning,
        ultrasonic_energy=ultrasonic_energy, ultrasonic_delta=ultrasonic_delta,
        noise_floor=noise_floor, scores=scores, frequencies=frequencies, times=times_decim,
        Sxx_db=Sxx_db, avg_spectrum_db=avg_db, nyquist=nyquist,
        evidence=evidence,
        edge_times=times_decim,
        edge_values=edges_all
    )

def fmt_time(s: float) -> str:
    if s < 60: return f"{s:.2f}s"
    return f"{int(s//60)}m {s%60:.1f}s"

def print_limitations():
    print("\n  Limitations:")
    print("    • This tool cannot prove lossless provenance.")
    print("    • High-bitrate MP3/AAC/Opus can resemble lossless in spectral analysis.")
    print("    • Some true lossless masters naturally lack high-frequency content.")
    print("    • Final confirmation requires trusted source metadata or AccurateRip/CUETools for CD rips.")

def build_json_report(res: SpectralAnalysisResult, container_codec: str, container_sr: str, container_bitrate: Optional[str]) -> dict:
    return {
        "verdict": res.evidence.verdict,
        "explanation": res.evidence.explanation,
        "container_codec": container_codec,
        "container_sample_rate_hz": int(container_sr) if container_sr.isdigit() else None,
        "container_bitrate": container_bitrate,
        "nyquist_hz": res.nyquist,
        "hard_cutoff": res.evidence.hard_cutoff,
        "strongest_drop_hz": res.evidence.best_drop_freq,
        "strongest_drop_db": res.evidence.max_drop_db,
        "active_edge_p10_hz": res.evidence.edge_p10,
        "active_edge_p50_hz": res.evidence.edge_p50,
        "active_edge_p90_hz": res.evidence.edge_p90,
        "cutoff_persistence": res.evidence.cutoff_persistence,
        "active_frames_pct": res.evidence.active_frames_pct,
        "high_band_db": res.evidence.high_band_db,
        "near_nyquist_db": res.evidence.near_nyquist_db,
        "sbr_likelihood": res.evidence.sbr_likelihood,
        "suspicious_windows": res.evidence.suspicious_windows,
        "closest_resemblance": res.profile,
        "resemblance_confidence": res.confidence,
        "flags": [
            {"severity": f.split(']')[0].strip('['), "text": f.split(']', 1)[1].strip() if ']' in f else f}
            for f in res.evidence.suspicious_flags
        ],
    }

def main():
    script_start = time.perf_counter()
    
    args = parser.parse_args()
    
    NPERSEG = 1024
    OVERLAP = 0.5
    DPI = 150
    FMT = "png"
    
    if args.file_path is None:
        args.file_path = input("Audio file: ").strip()
    
    file_path = args.file_path.strip().strip('"\'')
    if not os.path.isfile(file_path):
        print(f"Error: '{Path(file_path).name}' is not an audio file", file=sys.stderr)
        sys.exit(1)
    
    outputs_dir = Path(file_path).parent
    
    file_size = Path(file_path).stat().st_size
    print(f"\n{Path(file_path).name} ({file_size/1024/1024:.1f} MB)")
    
    if _import_time > 3:
        print("      (installed numpy/scipy, the next launches will be quicker)")
    
    # Audio signature checking
    audio_magic = {
        b'RIFF': 'WAV/AIFF',
        b'fLaC': 'FLAC',
        b'OggS': 'OGG/Opus/Vorbis',
        b'MThd': 'MIDI',
    }
    with open(file_path, 'rb') as f:
        head = f.read(16)
    ext = Path(file_path).suffix.lower()
    is_audio = any(head.startswith(m) for m in audio_magic) or ext in ('.wav', '.flac', '.ogg', '.opus', '.mp3', '.m4a', '.aac', '.wma', '.aiff', '.aif', '.ape', '.wv', '.mpc', '.dff', '.dsf', '.caf')
    if not is_audio:
        print(f"Error: '{Path(file_path).name}' does not appear to be an audio file", file=sys.stderr)
        sys.exit(1)

    print("[1/3] Loading...")
    t0 = time.perf_counter()
    data, sr = load_audio(file_path)
    load_time = time.perf_counter() - t0
    
    duration = len(data) / sr
    print(f"      {sr} Hz, {duration:.1f}s, {len(data):,} samples ({fmt_time(load_time)})")
    
    dynamics = analyze_dynamics(data, sr)
    print(f"      Peak: {dynamics.peak_db:.1f} dB | RMS: {dynamics.rms_db:.1f} dB | Crest: {dynamics.crest_factor:.1f} dB ({dynamics.dr_rating})")
    if dynamics.clip_percentage > 0:
        print(f"      ⚠ Clipping: {dynamics.clipped_samples:,} samples ({dynamics.clip_percentage:.3f}%)")
    
    data_display = data.mean(axis=1) if data.ndim > 1 else data
    
    if args.info:
        print(f"\n--- DYNAMICS ANALYSIS ---")
        print(f"  Peak level:      {dynamics.peak_db:.1f} dB")
        print(f"  RMS level:       {dynamics.rms_db:.1f} dB")
        print(f"  Crest factor:    {dynamics.crest_factor:.1f} dB")
        print(f"  Dynamic range:   {dynamics.dynamic_range:.1f} dB")
        print(f"  Rating:          {dynamics.dr_rating.upper()}")
        print(f"  Clipped samples: {dynamics.clipped_samples:,} ({dynamics.clip_percentage:.4f}%)")
        if len(dynamics.clip_times) > 0:
            times_str = ", ".join([f"{t:.2f}s" for t in dynamics.clip_times[:5]])
            print(f"  Clip locations:  {times_str}{'...' if len(dynamics.clip_times) > 5 else ''}")
        sys.exit(0)
    
    if args.detect:
        ffprobe_info = None
        try:
            result_probe = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path],
                capture_output=True, text=True, timeout=5
            )
            if result_probe.returncode == 0:
                ffprobe_info = json.loads(result_probe.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
        
        print("[2/3] Analyzing spectral evidence...")
        t0 = time.perf_counter()
        
        res = analyze_transcode_evidence(data, sr)
        detect_time = time.perf_counter() - t0
        
        print(f"\n{'='*50}")
        print("SPECTRAL ANALYSIS")
        print(f"{'='*50}")
        
        container_codec = "unknown"
        container_sr = str(sr)
        container_bitrate = None
        container_bit_depth = None
        if ffprobe_info:
            for stream in ffprobe_info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    container_codec = stream.get('codec_name', 'unknown')
                    container_sr = stream.get('sample_rate', str(sr))
                    container_bit_depth = stream.get('bits_per_raw_sample', stream.get('bits_per_sample', None))
                    if 'bit_rate' in stream:
                        container_bitrate = f"{int(stream['bit_rate'])//1000} kbps"
        
        print(f"  Container codec:   {container_codec.upper()}")
        print(f"  Sample rate:       {container_sr} Hz")
        if container_bit_depth:
            print(f"  Bit depth:         {container_bit_depth}-bit")
        if container_bitrate:
            print(f"  Bitrate:           {container_bitrate}")
        
        print(f"\n  Spectral verdict:  {res.evidence.verdict} - {res.evidence.explanation}")
        
        if res.evidence.verdict in ("WARN", "FAIL"):
            print(f"  Cutoff frequency:  {res.cutoff_freq:.0f} Hz")
        else:
            print(f"  Active edge (p50): {res.evidence.edge_p50:.0f} Hz")
        
        print(f"  Shelf type:        {res.shelf_type}")
        print(f"  SBR likelihood:    {res.evidence.sbr_likelihood}")
        
        print(f"\n  Evidence:")
        print(f"    Hard cutoff:            {'yes' if res.evidence.hard_cutoff else 'no'}")
        if res.evidence.hard_cutoff:
            print(f"    Strongest drop:         {res.evidence.best_drop_freq:.0f} Hz ({res.evidence.max_drop_db:.1f} dB)")
        print(f"    Active frames:          {res.evidence.active_frames_pct*100:.0f}%")
        print(f"    Cutoff persistence:     {res.evidence.cutoff_persistence*100:.0f}%")
        print(f"    High-band energy:       {res.evidence.high_band_db:.1f} dB")
        print(f"    Near-Nyquist energy:    {res.evidence.near_nyquist_db:.1f} dB")
        if sr > 48000 and res.ultrasonic_delta is not None:
            print(f"    Ultrasonic (24k+):      {res.ultrasonic_delta:.1f} dB above noise")
        
        if len(res.evidence.suspicious_flags) > 0:
            print(f"\n  Flags:")
            for flag in res.evidence.suspicious_flags:
                print(f"    • {flag}")
        
        if len(res.evidence.suspicious_windows) > 0:
            print(f"\n  Suspicious time windows:")
            for w in res.evidence.suspicious_windows[:8]:
                print(f"    • {w}")
            if len(res.evidence.suspicious_windows) > 8:
                print(f"    ... and {len(res.evidence.suspicious_windows) - 8} more")
        
        print(f"\n  Closest cutoff resemblance: {res.profile.upper()} ({res.confidence})")
        
        if res.transcode_warning:
            print(f"\n  ⚠ {res.transcode_warning}")
        
        print_limitations()
        
        print(f"{'='*50}")
        
        print("\nProfile resemblance scores:")
        sorted_scores = sorted(res.scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for codec, score in sorted_scores:
            print(f"  {codec:15} {score:3}/100")
        
        if args.json is not None:
            json_name = args.json if args.json else f"{Path(file_path).stem}_analysis.json"
            json_path = str(outputs_dir / json_name)
            report = build_json_report(res, container_codec, container_sr, container_bitrate)
            with open(json_path, 'w') as jf:
                json.dump(report, jf, indent=2)
            print(f"\n  JSON report saved: {json_path}")
        
        print(f"\n[3/3] Generating analysis plot...")
        t0 = time.perf_counter()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 11))
        ax1, ax2, ax3 = axes
        
        freqs = res.frequencies
        spectrum = res.avg_spectrum_db
        nyquist = res.nyquist
        
        ax1.plot(freqs, spectrum, 'b-', linewidth=0.8, alpha=0.7, label='Spectrum')
        ax1.axvline(x=res.cutoff_freq, color='r', linestyle='--', label=f"Cutoff: {res.cutoff_freq:.0f} Hz")
        if sr > 48000:
            ax1.axvline(x=24000, color='g', linestyle=':', alpha=0.5, label='24 kHz')
        ax1.axhline(y=res.noise_floor, color='gray', linestyle=':', alpha=0.5, label='Noise floor')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (dB)')
        
        if res.evidence.verdict == "FAIL":
            title = "Frequency Spectrum - Suspicious cutoff / possible lossy source"
        elif res.evidence.verdict == "WARN":
            title = "Frequency Spectrum - Suspicious cutoff / possible lossy source"
        elif res.evidence.verdict == "PASS":
            title = "Frequency Spectrum - No obvious lossy transcode signature"
        else:
            title = "Frequency Spectrum - Inconclusive spectral evidence"
        ax1.set_title(title)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, nyquist)
        
        extent = [res.times[0], res.times[-1], freqs[0], freqs[-1]]
        im = ax2.imshow(res.Sxx_db, aspect='auto', origin='lower', extent=extent, cmap='inferno', interpolation='bilinear')
        ax2.axhline(y=res.cutoff_freq, color='white', linestyle='--', alpha=0.7)
        if sr > 48000:
            ax2.axhline(y=24000, color='green', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Spectrogram')
        ax2.set_ylim(0, nyquist)
        plt.colorbar(im, ax=ax2, label='dB')
        
        edge_times = res.edge_times
        edge_values = res.edge_values
        valid = ~np.isnan(edge_values)
        ax3.plot(edge_times[valid], edge_values[valid], 'c-', linewidth=0.6, alpha=0.8, label='Active edge')
        ax3.axhline(y=res.nyquist * 0.94, color='gray', linestyle=':', alpha=0.5, label='94% Nyquist')
        if res.evidence.hard_cutoff:
            ax3.axhline(y=res.evidence.best_drop_freq, color='r', linestyle='--', alpha=0.6, label=f"Best drop: {res.evidence.best_drop_freq:.0f} Hz")
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Active Spectral Edge Over Time')
        ax3.set_ylim(0, nyquist)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = args.output if args.output else str(outputs_dir / f"{Path(file_path).stem}_analysis.png")
        plt.savefig(output_path, dpi=DPI, format=FMT)
        
        print(f"      Saved: {output_path}")
        if not args.no_open:
            open_file(output_path)
        
        total = time.perf_counter() - script_start
        print(f"\nDone in {fmt_time(total)}")
        sys.exit(0)
    
    if args.compare:
        if not os.path.isfile(args.compare):
            print(f"Error: Comparison file not found: {args.compare}", file=sys.stderr)
            sys.exit(1)
        
        print(f"[2/4] Loading comparison file...")
        data2, sr2 = load_audio(args.compare)
        data2_display = data2.mean(axis=1) if data2.ndim > 1 else data2
        
        if sr2 != sr:
            data2_display = resample(data2_display, int(len(data2_display) * sr / sr2))
            print(f"      Resampled {sr2} -> {sr} Hz")
        
        align_len = min(sr * 5, len(data_display), len(data2_display))
        corr = correlate(data_display[:align_len], data2_display[:align_len], mode='full')
        delay = np.argmax(np.abs(corr)) - (align_len - 1)
        
        if delay > 0:
            print(f"      Aligned: delayed B by {delay/sr*1000:.1f}ms")
            data2_display = data2_display[delay:]
        elif delay < 0:
            print(f"      Aligned: delayed A by {-delay/sr*1000:.1f}ms")
            data_display = data_display[-delay:]
            
        min_len = min(len(data_display), len(data2_display))
        data_display = data_display[:min_len]
        data2_display = data2_display[:min_len]
        
        diff = data_display - data2_display
        diff_rms = np.sqrt(np.mean(diff**2))
        corr = np.corrcoef(data_display, data2_display)[0, 1] if len(data_display) > 1 else 0
        similarity = max(0, corr) * 100
        
        print(f"      Analyzing {min_len/sr:.2f}s")
        print(f"      Similarity: {similarity:.1f}%")
        print(f"      Difference RMS: {20*np.log10(diff_rms + 1e-10):.1f} dB")
        
        print("[3/4] Computing spectrograms...")
        t0 = time.perf_counter()
        
        nperseg = 1024
        noverlap = 512
        
        frequencies, times, Z1 = stft(data_display, fs=sr, nperseg=nperseg, noverlap=noverlap)
        _, _, Z2 = stft(data2_display, fs=sr, nperseg=nperseg, noverlap=noverlap)
        Zd = Z1 - Z2
        
        time_decim = max(1, Z1.shape[1] // 2000)
        S1_db = 10 * np.log10(np.abs(Z1[:, ::time_decim])**2 + 1e-10)
        S2_db = 10 * np.log10(np.abs(Z2[:, ::time_decim])**2 + 1e-10)
        Sdiff_db = 10 * np.log10(np.abs(Zd[:, ::time_decim])**2 + 1e-10)
        times = times[::time_decim]
        
        stft_time = time.perf_counter() - t0
        print(f"      Done ({fmt_time(stft_time)})")
        
        print("[4/4] Rendering...")
        t0 = time.perf_counter()
        
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
        output_path = args.output if args.output else str(outputs_dir / f"{Path(file_path).stem}_vs_{Path(args.compare).stem}.png")
        plt.savefig(output_path, dpi=DPI, format=FMT)
        
        print(f"      Saved: {output_path}")
        if not args.no_open:
            open_file(output_path)
        
        total = time.perf_counter() - script_start
        print(f"\nDone in {fmt_time(total)}")
        sys.exit(0)
    
    
    print("[2/3] Computing STFT...")
    t0 = time.perf_counter()
    
    noverlap = int(NPERSEG * OVERLAP)
    frequencies, times, Zxx = stft(data_display, fs=sr, nperseg=NPERSEG, noverlap=noverlap, window='hann')
    
    time_decimation = max(1, Zxx.shape[1] // 2000)
    Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decimation]) ** 2 + 1e-10)
    times = times[::time_decimation]
    
    stft_time = time.perf_counter() - t0
    print(f"      {Sxx_db.shape[0]}x{Sxx_db.shape[1]} bins, {sr/NPERSEG:.1f} Hz res ({fmt_time(stft_time)})")
    
    print("[3/3] Rendering...")
    t0 = time.perf_counter()
    
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
    
    output_path = args.output if args.output else str(outputs_dir / f"{Path(file_path).stem}.png")
    
    t0 = time.perf_counter()
    plt.savefig(output_path, dpi=DPI, format=FMT)
    save_time = time.perf_counter() - t0
    
    print(f"      Saved: {output_path} ({fmt_time(save_time)})")
    
    if not args.no_open:
        open_file(output_path)
    
    plt.close('all')
    
    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)} (load:{fmt_time(load_time)} stft:{fmt_time(stft_time)} render:{fmt_time(render_time)} save:{fmt_time(save_time)})")

if __name__ == "__main__":
    main()
