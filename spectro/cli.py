import argparse
import os
import sys
import time
from pathlib import Path

from .audio import looks_like_audio

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

DEPS = ("numpy", "scipy", "soundfile", "matplotlib")


def print_version() -> None:
    import platform
    import shutil
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    from . import __version__

    def v(name: str) -> str:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "?"

    print(VERSION_ART)
    print(f"    spectro v{__version__} - spectrograms + lossy-source forensics")
    print(f"    python {platform.python_version()} · "
          + " · ".join(f"{d} {v(d)}" for d in DEPS))
    tools = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t)]
    print(f"    {' + '.join(tools) if tools else 'no ffmpeg'}"
          f"{'' if tools else ' (mp3/m4a decoding and container info unavailable)'}")
    print()


def print_intro() -> None:
    print("\nspectro - audio spectrograms + lossy-source detection\n")
    print('  spectro "song.flac"               render spectrogram')
    print('  spectro "song.flac" --detect      lossy-source forensics')
    print('  spectro "song.flac" --preview     spectrogram in the terminal')
    print('  spectro a.flac --compare b.flac   null-test comparison')
    print('  spectro --help                    everything else\n')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spectro",
        description="Spectrograms and heuristic lossy-source analysis for audio files.")
    p.add_argument("file_path", nargs="*", help="Input audio file(s)")
    p.add_argument("-o", "--output", help="Output filename")
    p.add_argument("--detect", action="store_true", help="Spectral authenticity / transcode-evidence analysis")
    p.add_argument("-p", "--preview", action="store_true", help="Render spectrogram in the terminal (no PNG)")
    p.add_argument("--compare", metavar="FILE", help="Compare with another audio file")
    p.add_argument("--log", action="store_true", help="Log frequency axis")
    p.add_argument("--no-open", dest="no_open", action="store_true", help="Don't auto-open output file")
    p.add_argument("--info", action="store_true", help="Show file info only")
    p.add_argument("--json", nargs="?", const="", metavar="FILE", help="Write a JSON report")
    p.add_argument("--verbose", action="store_true", help="Print per-feature subscores in detect mode")
    p.add_argument("--fails-only", dest="fails_only", action="store_true",
                   help="Batch detect: only table the WARNs and FAILs")
    p.add_argument("--sort", action="store_true",
                   help="Batch detect: worst first instead of input order")
    p.add_argument("-j", "--jobs", type=int, metavar="N",
                   help="Batch detect: files to analyze at once (default: 4)")
    p.add_argument("-v", "--version", action="store_true", help="Show version info")
    return p


def resolve_path(raw: str) -> str:
    """Tidy up whatever the shell or a drag-and-drop left us with."""
    path = raw.strip().strip('"\'')
    if not os.path.isfile(path) and '\\' in path:
        # dropping a file onto a macOS terminal escapes the spaces
        unescaped = path.replace('\\ ', ' ').replace('\\\\', '\\')
        if os.path.isfile(unescaped):
            return unescaped
    return path


# first one asked for wins
MODES = ("detect", "compare", "preview", "info")


def pick_mode(args) -> str:
    return next((m for m in MODES if getattr(args, m)), "basic")


def warn_ignored(args) -> None:
    asked = [f"--{m}" for m in MODES if getattr(args, m)]
    if len(asked) > 1:
        print(f"Note: {asked[0]} wins, ignoring {', '.join(asked[1:])}", file=sys.stderr)
    if args.json is not None and not args.detect:
        print("Note: --json only writes anything in --detect mode", file=sys.stderr)


def run_one(args, file_path: str, script_start: float) -> None:
    outputs_dir = Path(file_path).parent

    file_size = Path(file_path).stat().st_size
    print(f"\n{Path(file_path).name} ({file_size/1024/1024:.1f} MB)")

    mode = pick_mode(args)
    cold_hint = (" (a cold start can take 10+ seconds on some systems)"
                 if mode in ("detect", "compare") else "")
    print(f"[0/3] Starting analysis tools...{cold_hint}", flush=True)
    tools_start = time.perf_counter()

    from .reporting import fmt_time

    if mode == "detect":
        from .commands.detect import cmd_detect as command
    elif mode == "compare":
        from .commands.compare import cmd_compare as command
    elif mode == "preview":
        from .commands.preview import cmd_preview as command
    elif mode == "info":
        from .commands.info import cmd_info as command
    else:
        from .commands.basic import cmd_basic as command

    if mode in ("detect", "compare"):
        from .audio import load_audio
        from .dynamics import analyze_dynamics
    elif mode == "info":
        from .dynamics import analyze_dynamics_file

    print(f"      Ready ({fmt_time(time.perf_counter() - tools_start)})")

    if mode in ("preview", "basic"):
        try:
            if mode == "preview":
                command(file_path, script_start)
            else:
                command(args, file_path, outputs_dir, script_start)
        except ValueError as ex:
            if str(ex) != "audio has no samples":
                raise
            print(f"Error: '{Path(file_path).name}' contains no audio samples",
                  file=sys.stderr)
        return
    if mode == "info":
        print("[1/3] Scanning audio once for exact dynamics...", flush=True)
        t0 = time.perf_counter()
        dynamics, sr, frame_count, channels = analyze_dynamics_file(file_path)
        scan_time = time.perf_counter() - t0
        if frame_count == 0:
            print(f"Error: '{Path(file_path).name}' contains no audio samples",
                  file=sys.stderr)
            return
        duration = frame_count / sr
        print(f"      {sr} Hz, {duration:.1f}s, {frame_count:,} samples "
              f"({fmt_time(scan_time)})")
        print(f"      Peak: {dynamics.peak_db:.1f} dB | RMS: {dynamics.rms_db:.1f} dB | "
              f"Crest: {dynamics.crest_factor:.1f} dB ({dynamics.dr_rating})")
        command(dynamics, sr, duration, channels)
        return

    print("[1/3] Loading...")
    t0 = time.perf_counter()
    data, sr = load_audio(file_path)
    load_time = time.perf_counter() - t0

    if len(data) == 0:
        print(f"Error: '{Path(file_path).name}' contains no audio samples", file=sys.stderr)
        return

    duration = len(data) / sr
    print(f"      {sr} Hz, {duration:.1f}s, {len(data):,} samples ({fmt_time(load_time)})")

    dynamics = analyze_dynamics(data, sr)
    print(f"      Peak: {dynamics.peak_db:.1f} dB | RMS: {dynamics.rms_db:.1f} dB | Crest: {dynamics.crest_factor:.1f} dB ({dynamics.dr_rating})")
    if dynamics.clip_percentage > 0:
        print(f"      ⚠ Clipping: {dynamics.clipped_samples:,} samples ({dynamics.clip_percentage:.3f}%)")

    if mode == "detect":
        command(args, file_path, data, sr, outputs_dir, script_start)
        return
    if mode == "compare":
        data_display = data.mean(axis=1) if data.ndim > 1 else data
        command(args, file_path, data_display, sr, outputs_dir, script_start)


def run(script_start: float) -> None:
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

    warn_ignored(args)
    paths = [resolve_path(p) for p in args.file_path]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"Error: '{Path(p).name}' not found", file=sys.stderr)
        sys.exit(1)

    if len(paths) > 1:
        if args.detect:
            print("\nStarting analysis tools... (a cold start can take 10+ seconds on some systems)",
                  flush=True)
            from .commands.batch import cmd_batch
            cmd_batch(args, paths, script_start)
            return
        if args.output:
            print("Error: -o takes one output name, so it needs one input file", file=sys.stderr)
            sys.exit(1)
        for p in paths:
            if looks_like_audio(Path(p)):
                run_one(args, p, time.perf_counter())
            else:
                print(f"\nSkipping '{Path(p).name}', doesn't look like audio")
        return

    file_path = paths[0]
    if not looks_like_audio(Path(file_path)):
        print(f"Error: '{Path(file_path).name}' does not appear to be an audio file", file=sys.stderr)
        sys.exit(1)

    run_one(args, file_path, script_start)


def main():
    script_start = time.perf_counter()
    try:
        run(script_start)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        os._exit(0)


if __name__ == "__main__":
    main()
