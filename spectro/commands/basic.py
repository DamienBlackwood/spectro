import time
from pathlib import Path

from ..audio import open_file
from ..dataclasses_ import T
from ..display import analyze_display, compute_display_spectrogram
from ..plotting import lazy_pyplot
from ..reporting import fmt_time


def cmd_basic(args, file_path: str, outputs_dir: Path, script_start: float) -> None:
    print("[1/3] Scanning audio once (exact dynamics + display frames)...", flush=True)
    t0 = time.perf_counter()
    frames = analyze_display(file_path, max_time_bins=2000)
    scan_time = time.perf_counter() - t0

    dynamics = frames.dynamics
    print(f"      {frames.sample_rate} Hz, {frames.duration:.1f}s, "
          f"{frames.frame_count:,} samples ({fmt_time(scan_time)})")
    print(f"      Peak: {dynamics.peak_db:.1f} dB | RMS: {dynamics.rms_db:.1f} dB | "
          f"Crest: {dynamics.crest_factor:.1f} dB ({dynamics.dr_rating})")
    if dynamics.clip_percentage > 0:
        print(f"      ⚠ Clipping: {dynamics.clipped_samples:,} samples "
              f"({dynamics.clip_percentage:.3f}%)")

    print("[2/3] Computing retained STFT frames...")
    t0 = time.perf_counter()
    spec = compute_display_spectrogram(frames)
    stft_time = time.perf_counter() - t0

    print(f"      {spec.Sxx_db.shape[0]}x{spec.Sxx_db.shape[1]} bins, "
          f"{spec.frequency_resolution:.1f} Hz res; skipped "
          f"{frames.full_stft_frames - len(frames.windows):,} discarded frames "
          f"({fmt_time(stft_time)})")

    print("[3/3] Rendering...")
    t0 = time.perf_counter()

    plt = lazy_pyplot(Path(file_path).stem)
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [spec.times[0], spec.times[-1],
              spec.frequencies[0], spec.frequencies[-1]]
    im = ax.imshow(spec.Sxx_db, aspect='auto', origin='lower', extent=extent,
                   cmap='inferno', interpolation='bilinear')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'{Path(file_path).stem}')
    plt.colorbar(im, ax=ax, label='dB')

    if args.log:
        ax.set_yscale('log')
        ax.set_ylim(20, spec.frequencies[-1])

    plt.tight_layout()
    render_time = time.perf_counter() - t0

    output_path = args.output if args.output else str(outputs_dir / f"{Path(file_path).stem}.png")

    t0 = time.perf_counter()
    plt.savefig(output_path, dpi=T.dpi, format=T.output_fmt)
    save_time = time.perf_counter() - t0

    print(f"      Saved: {output_path} ({fmt_time(save_time)})")

    if not args.no_open:
        open_file(output_path)

    plt.close(fig)

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)} (scan:{fmt_time(scan_time)} "
          f"fft:{fmt_time(stft_time)} render:{fmt_time(render_time)} "
          f"save:{fmt_time(save_time)})")
