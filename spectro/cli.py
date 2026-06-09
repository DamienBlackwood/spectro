# time the heavy imports so we can hint on cold starts
import time as _t
_import_start = _t.perf_counter()

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

_import_time = _t.perf_counter() - _import_start

VERSION_ART = r"""
                ░▒▓█ spectro █▓▒░
      ┌──────────────────────────────────┐
    ↑ │      ·   ·       ·    ·      ·   │ ← nothing up here? sus.
    f │ ╌╌╌╌╌╌╌╌╌╌╌╌ cutoff ╌╌╌╌╌╌╌╌╌╌╌╌ │
    r │ ░▒▓▒░▒▒▓▒▒░░▒▒▓▒░▒▒▒▓▒░▒▒░▒▓▒▒░▒ │
    e │ ▓▓█▓▓▒▓▓▓█▓▓▓▒▓▓███▓▓▒▓▓▓█▓▓▓▒▓▓ │
    q │ ████████████████████████████████ │
      └──────────────────────────────────┘
                     time →
"""


def print_version() -> None:
    import platform
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    from . import __version__

    def v(name: str) -> str:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "?"

    print(VERSION_ART)
    print(f"    spectro v{__version__} - spectrograms + lossy-source forensics")
    print(f"    python {platform.python_version()} · numpy {v('numpy')} · scipy {v('scipy')}"
          f" · soundfile {v('soundfile')} · matplotlib {v('matplotlib')}")
    print()


def print_intro() -> None:
    print("\nspectro - audio spectrograms + lossy-source detection\n")
    print('  spectro "song.flac"               render spectrogram')
    print('  spectro "song.flac" --detect      lossy-source forensics')
    print('  spectro a.flac --compare b.flac   null-test comparison')
    print('  spectro --help                    everything else\n')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spectrogram Generator")
    p.add_argument("file_path", nargs="*", help="Input audio file(s)")
    p.add_argument("-o", "--output", help="Output filename")
    p.add_argument("--detect", action="store_true", help="Spectral authenticity / transcode-evidence analysis")
    p.add_argument("-p", "--preview", action="store_true", help="Render spectrogram in the terminal (no PNG)")
    p.add_argument("--compare", metavar="FILE", help="Compare with another audio file")
    p.add_argument("--log", action="store_true", help="Log frequency axis")
    p.add_argument("--no-open", dest="no_open", action="store_true", help="Don't auto-open output file")
    p.add_argument("--info", action="store_true", help="Show file info only")
    p.add_argument("--json", nargs="?", const="", metavar="FILE", help="Write machine-readable JSON report")
    p.add_argument("--verbose", action="store_true", help="Print per-feature subscores in detect mode")
    p.add_argument("-v", "--version", action="store_true", help="Show version info")
    return p


def main():
    script_start = time.perf_counter()

    args = build_parser().parse_args()

    if args.version:
        print_version()
        return

    if not args.file_path:
        print_intro()
        try:
            entered = input("Audio file (or just hit enter to bail): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not entered:
            return
        args.file_path = [entered]

    if len(args.file_path) > 1:
        if args.detect:
            from .commands.batch import cmd_batch
            cmd_batch(args, args.file_path, script_start)
        else:
            print("Multiple files only work in --detect batch mode for now, e.g.:")
            print("  spectro *.flac --detect")
            sys.exit(1)
        return

    file_path = args.file_path[0].strip().strip('"\'')
    if not os.path.isfile(file_path):
        print(f"Error: '{Path(file_path).name}' not found", file=sys.stderr)
        sys.exit(1)

    if not looks_like_audio(Path(file_path)):
        print(f"Error: '{Path(file_path).name}' does not appear to be an audio file", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path(file_path).parent

    file_size = Path(file_path).stat().st_size
    print(f"\n{Path(file_path).name} ({file_size/1024/1024:.1f} MB)")

    if _import_time > 3:
        print("      (cold start, next launches will be quicker!)")

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

    if args.preview:
        from .commands.preview import cmd_preview
        cmd_preview(data_display, sr, script_start)
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
