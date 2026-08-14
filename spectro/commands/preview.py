import time

from ..display import analyze_display, compute_display_spectrogram
from ..reporting import fmt_time
from ..term import render_term_spectrogram, spectrogram_width


def cmd_preview(file_path: str, script_start: float) -> None:
    width = spectrogram_width()

    print("[1/3] Scanning audio once (exact dynamics + display frames)...", flush=True)
    t0 = time.perf_counter()
    frames = analyze_display(file_path, max_time_bins=4000)
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
    spec = compute_display_spectrogram(frames, preview_floor=True)
    fft_time = time.perf_counter() - t0
    print(f"      {spec.Sxx_db.shape[0]}x{spec.Sxx_db.shape[1]} bins, "
          f"{spec.frequency_resolution:.1f} Hz res; skipped "
          f"{frames.full_stft_frames - len(frames.windows):,} discarded frames "
          f"({fmt_time(fft_time)})")

    print("[3/3] Rendering terminal preview...")
    t0 = time.perf_counter()
    print()
    render_term_spectrogram(spec.Sxx_db, spec.frequencies,
                            duration=frames.duration, width=width)
    render_time = time.perf_counter() - t0

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)} (scan:{fmt_time(scan_time)} "
          f"fft:{fmt_time(fft_time)} render:{fmt_time(render_time)})")
