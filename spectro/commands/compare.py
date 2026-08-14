import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import correlate, resample_poly, stft

from ..audio import load_audio, open_file
from ..dataclasses_ import T
from ..plotting import lazy_pyplot
from ..reporting import fmt_time
from ..spectral import stft_params

BANDS = [(20, 250, "low"), (250, 2000, "mid"), (2000, 8000, "high"), (8000, None, "air")]


def band_deltas(S1_db: np.ndarray, S2_db: np.ndarray, freqs: np.ndarray) -> list:
    """Mean level difference per band, B relative to A."""
    out = []
    for low, high, name in BANDS:
        mask = freqs >= low if high is None else (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        out.append((name, float(np.mean(S2_db[mask]) - np.mean(S1_db[mask]))))
    return out


def _align_signals(a: np.ndarray, b: np.ndarray, sample_rate: int,
                   seconds: float = 5.0):
    """Trim the leading signal so the first few seconds line up."""
    align_len = min(int(sample_rate * seconds), len(a), len(b))
    if align_len < 2:
        return a, b, 0
    corr = correlate(a[:align_len], b[:align_len], mode='full')
    magnitude = np.abs(corr)
    if not np.all(np.isfinite(magnitude)) or float(np.max(magnitude)) <= 1e-12:
        return a, b, 0
    lag = int(np.argmax(magnitude)) - (align_len - 1)
    if 0 < lag < len(a):
        a = a[lag:]
    elif -len(b) < lag < 0:
        b = b[-lag:]
    return a, b, lag


def cmd_compare(args, file_path: str, data_display: np.ndarray, sr: int,
                outputs_dir: Path, script_start: float) -> None:
    if not os.path.isfile(args.compare):
        print(f"Error: Comparison file not found: {args.compare}", file=sys.stderr)
        sys.exit(1)

    print("[2/4] Loading comparison file...")
    data2, sr2 = load_audio(args.compare)
    data2_display = data2.mean(axis=1) if data2.ndim > 1 else data2

    if sr2 != sr:
        ratio = Fraction(sr, sr2).limit_denominator(1000)
        data2_display = resample_poly(data2_display, ratio.numerator, ratio.denominator)
        print(f"      Resampled {sr2} -> {sr} Hz")

    data_display, data2_display, delay = _align_signals(data_display, data2_display, sr)
    if delay > 0:
        print(f"      Aligned: B leads A by {delay/sr*1000:.1f}ms")
    elif delay < 0:
        print(f"      Aligned: B lags A by {-delay/sr*1000:.1f}ms")

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

    nperseg, noverlap = stft_params(min_len, T.display_nperseg)

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

    deltas = band_deltas(S1_db, S2_db, frequencies)
    print("      B vs A: " + "  ".join(f"{n} {d:+.1f} dB" for n, d in deltas))

    print("[4/4] Rendering...")

    plt = lazy_pyplot()
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
    plt.savefig(output_path, dpi=T.dpi, format=T.output_fmt)
    plt.close(fig)

    print(f"      Saved: {output_path}")
    if not args.no_open:
        open_file(output_path)

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
