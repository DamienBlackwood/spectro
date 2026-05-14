import argparse
import os
import sys
import time
from pathlib import Path

from .audio import load_audio, looks_like_audio
from .commands.basic import cmd_basic
from .commands.compare import cmd_compare
from .commands.detect import cmd_detect
from .commands.info import cmd_info
from .dynamics import analyze_dynamics
from .reporting import fmt_time

# Track cold-import cost to surface install hint
import time as _t
_import_start = _t.perf_counter()
import numpy as _np  
import soundfile as _sf 
_import_time = _t.perf_counter() - _import_start


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spectrogram Generator")
    p.add_argument("file_path", nargs="?", help="Input audio file")
    p.add_argument("-o", "--output", help="Output filename")
    p.add_argument("--detect", action="store_true", help="Spectral authenticity / transcode-evidence analysis")
    p.add_argument("--compare", metavar="FILE", help="Compare with another audio file")
    p.add_argument("--log", action="store_true", help="Log frequency axis")
    p.add_argument("--no-open", dest="no_open", action="store_true", help="Don't auto-open output file")
    p.add_argument("--info", action="store_true", help="Show file info only")
    p.add_argument("--json", nargs="?", const=None, metavar="FILE", help="Write machine-readable JSON report")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-feature subscores in detect mode")
    return p


def main():
    script_start = time.perf_counter()

    args = build_parser().parse_args()

    if args.file_path is None:
        args.file_path = input("Audio file: ").strip()

    file_path = args.file_path.strip().strip('"\'')
    if not os.path.isfile(file_path):
        print(f"Error: '{Path(file_path).name}' is not an audio file", file=sys.stderr)
        sys.exit(1)

    if not looks_like_audio(Path(file_path)):
        print(f"Error: '{Path(file_path).name}' does not appear to be an audio file", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path(file_path).parent

    file_size = Path(file_path).stat().st_size
    print(f"\n{Path(file_path).name} ({file_size/1024/1024:.1f} MB)")

    if _import_time > 3:
        print("      (installed numpy/scipy, the next launches will be quicker!)")

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
        cmd_info(dynamics)
        return

    if args.detect:
        cmd_detect(args, file_path, data, sr, outputs_dir, script_start)
        return

    if args.compare:
        cmd_compare(args, file_path, data_display, sr, outputs_dir, script_start)
        return

    cmd_basic(args, file_path, data_display, sr, outputs_dir, script_start, load_time)


if __name__ == "__main__":
    main()
