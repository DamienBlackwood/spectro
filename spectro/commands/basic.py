import time
from pathlib import Path

import numpy as np
from scipy.signal import stft

from ..audio import open_file
from ..dataclasses_ import T
from ..plotting import lazy_pyplot
from ..reporting import fmt_time


def cmd_basic(args, file_path: str, data_display: np.ndarray, sr: int,
              outputs_dir: Path, script_start: float, load_time: float) -> None:
    print("[2/3] Computing STFT...")
    t0 = time.perf_counter()

    noverlap = int(T.display_nperseg * T.display_overlap)
    frequencies, times, Zxx = stft(data_display, fs=sr, nperseg=T.display_nperseg, noverlap=noverlap, window='hann')

    time_decimation = max(1, Zxx.shape[1] // 2000)
    Sxx_db = 10 * np.log10(np.abs(Zxx[:, ::time_decimation]) ** 2 + 1e-10)
    times = times[::time_decimation]

    stft_time = time.perf_counter() - t0
    print(f"      {Sxx_db.shape[0]}x{Sxx_db.shape[1]} bins, {sr/T.display_nperseg:.1f} Hz res ({fmt_time(stft_time)})")

    print("[3/3] Rendering...")
    t0 = time.perf_counter()

    plt = lazy_pyplot()
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
    plt.savefig(output_path, dpi=T.dpi, format=T.output_fmt)
    save_time = time.perf_counter() - t0

    print(f"      Saved: {output_path} ({fmt_time(save_time)})")

    if not args.no_open:
        open_file(output_path)

    plt.close('all')

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)} (load:{fmt_time(load_time)} stft:{fmt_time(stft_time)} render:{fmt_time(render_time)} save:{fmt_time(save_time)})")
