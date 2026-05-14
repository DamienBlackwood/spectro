import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ..audio import open_file
from ..dataclasses_ import T
from ..plotting import lazy_pyplot
from ..reporting import build_json_report, fmt_time, print_limitations
from ..spectral import analyze_transcode_evidence


def _probe_container(file_path: str):
    """Best-effort ffprobe metadata extraction. Returns (codec, sr, bitrate, bit_depth)."""
    container_codec = "unknown"
    container_sr = None
    container_bitrate = None
    container_bit_depth = None

    ffprobe_info = None
    try:
        result_probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', '-show_streams', file_path],
            capture_output=True, text=True, timeout=5
        )
        if result_probe.returncode == 0:
            ffprobe_info = json.loads(result_probe.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    if ffprobe_info:
        for stream in ffprobe_info.get('streams', []):
            if stream.get('codec_type') == 'audio':
                container_codec = stream.get('codec_name', 'unknown')
                container_sr = stream.get('sample_rate')
                container_bit_depth = stream.get('bits_per_raw_sample', stream.get('bits_per_sample'))
                if 'bit_rate' in stream:
                    container_bitrate = f"{int(stream['bit_rate'])//1000} kbps"
                break

    return container_codec, container_sr, container_bitrate, container_bit_depth


def cmd_detect(args, file_path: str, data: np.ndarray, sr: int,
               outputs_dir: Path, script_start: float) -> None:
    container_codec, container_sr_raw, container_bitrate, container_bit_depth = _probe_container(file_path)
    container_sr = container_sr_raw if container_sr_raw else str(sr)

    print("[2/3] Analyzing spectral evidence...")
    res = analyze_transcode_evidence(data, sr)

    print(f"\n{'='*50}")
    print("SPECTRAL ANALYSIS")
    print(f"{'='*50}")

    print(f"  Container codec:   {container_codec.upper()}")
    print(f"  Sample rate:       {container_sr} Hz")
    if container_bit_depth:
        print(f"  Bit depth:         {container_bit_depth}-bit")
    if container_bitrate:
        print(f"  Bitrate:           {container_bitrate}")

    print(f"\n  Spectral verdict:  {res.evidence.verdict} - {res.evidence.explanation}")
    print(f"  Lossy evidence:    {res.evidence.lossy_score:.0f}/100")
    print(f"  Data quality:      {res.evidence.quality_score:.0f}/100")

    print(f"\n  Active edge:       p10={res.evidence.edge_p10:.0f}  p50={res.evidence.edge_p50:.0f}  p90={res.evidence.edge_p90:.0f} Hz")
    print(f"  Edge jitter:       {res.evidence.edge_jitter_hz:.0f} Hz (MAD)")
    print(f"  Rolloff variance:  {res.evidence.rolloff_85_var_hz:.0f} Hz (std)")
    print(f"  Filter slope:      {res.evidence.max_slope_db_per_khz:.1f} dB/kHz")
    print(f"  Band ratio (H/L):  {res.evidence.band_ratio_db:.1f} dB")
    print(f"  SBR likelihood:    {res.evidence.sbr_likelihood}")
    if res.evidence.hard_cutoff:
        print(f"  Strongest drop:    {res.evidence.best_drop_freq:.0f} Hz ({res.evidence.max_drop_db:.1f} dB)")
    print(f"  Cutoff persistence: {res.evidence.cutoff_persistence*100:.0f}%  (active frames: {res.evidence.active_frames_pct*100:.0f}%)")
    if sr > T.high_sample_rate_hz and res.ultrasonic_delta is not None:
        print(f"  Ultrasonic (24k+): {res.ultrasonic_delta:.1f} dB above noise")

    if args.verbose and res.evidence.subscores:
        print(f"\n  Subscores:")
        for name, val in sorted(res.evidence.subscores.items(), key=lambda x: -x[1]):
            print(f"    {name:14} {val:5.1f}/100")

    if len(res.evidence.suspicious_flags) > 0:
        print(f"\n  Flags:")
        for flag in res.evidence.suspicious_flags:
            print(f"    • [{flag.severity.upper()}] {flag.name}: {flag.detail}")

    if len(res.evidence.suspicious_windows) > 0:
        print(f"\n  Suspicious time windows:")
        for w in res.evidence.suspicious_windows[:8]:
            print(f"    • {w}")
        if len(res.evidence.suspicious_windows) > 8:
            print(f"    ... and {len(res.evidence.suspicious_windows) - 8} more")

    print(f"\n  Closest cutoff resemblance: {res.profile.upper()}")

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
        report = build_json_report(res, container_codec, container_sr, container_bitrate, container_bit_depth)
        with open(json_path, 'w') as jf:
            json.dump(report, jf, indent=2)
        print(f"\n  JSON report saved: {json_path}")

    print(f"\n[3/3] Generating analysis plot...")
    plt = lazy_pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    ax1, ax2, ax3 = axes

    freqs = res.frequencies
    spectrum = res.avg_spectrum_db
    nyquist = res.nyquist

    ax1.plot(freqs, spectrum, 'b-', linewidth=0.8, alpha=0.7, label='Spectrum')
    ax1.axvline(x=res.cutoff_freq, color='r', linestyle='--', label=f"Cutoff: {res.cutoff_freq:.0f} Hz")
    if sr > T.high_sample_rate_hz:
        ax1.axvline(x=24000, color='g', linestyle=':', alpha=0.5, label='24 kHz')
    ax1.axhline(y=res.noise_floor, color='gray', linestyle=':', alpha=0.5, label='Noise floor')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')

    if res.evidence.verdict == "FAIL":
        title = "Frequency Spectrum - Strong lossy transcode evidence"
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
    if sr > T.high_sample_rate_hz:
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
    ax3.axhline(y=res.nyquist * T.nyquist_persistence_default, color='gray', linestyle=':', alpha=0.5, label='94% Nyquist')
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
    plt.savefig(output_path, dpi=T.dpi, format=T.output_fmt)

    print(f"      Saved: {output_path}")
    if not args.no_open:
        open_file(output_path)

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
