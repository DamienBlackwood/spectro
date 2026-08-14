#!/usr/bin/env python3
"""Build a transcode corpus from your own lossless files, then score it.

    python tests/make_corpus.py ~/Music/*.flac

Takes each file, runs it through ffmpeg at a spread of bitrates, decodes the
result back to FLAC and asks spectro what it thinks. The originals should never
come back WARN/FAIL and the transcodes should. INCONCLUSIVE is reported on its
own: it is not a false positive and it is not a caught transcode.

Nothing gets written next to your music, it all lands in a temp dir.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectro.audio import load_audio                    # noqa: E402
from spectro.spectral import analyze_transcode_evidence  # noqa: E402

ENCODES = [
    ("mp3_128", ["-c:a", "libmp3lame", "-b:a", "128k"], ".mp3"),
    ("vorbis_q4", ["-c:a", "vorbis", "-strict", "-2", "-q:a", "4"], ".ogg"),
    ("opus_128", ["-c:a", "libopus", "-b:a", "128k"], ".opus"),
    ("mp3_320", ["-c:a", "libmp3lame", "-b:a", "320k"], ".mp3"),
    ("aac_192", ["-c:a", "aac", "-b:a", "192k"],        ".m4a"),
    ("apple_aac_128", ["-c:a", "aac_at", "-b:a", "128k"], ".m4a"),
    ("mp3_v0",  ["-c:a", "libmp3lame", "-q:a", "0"],    ".mp3"),
    ("apple_aac_256", ["-c:a", "aac_at", "-b:a", "256k"], ".m4a"),
]


def ffmpeg(args) -> bool:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-vn"] + args,
                       capture_output=True)
    return r.returncode == 0


def score(path: Path):
    data, sr = load_audio(str(path))
    e = analyze_transcode_evidence(data, sr).evidence
    return sr, e.verdict, e.lossy_score, e.cutoff_freq, e.shelf_type


def main(sources):
    if not shutil.which("ffmpeg"):
        sys.exit("Need ffmpeg on PATH to build the corpus.")

    clean_total = clean_flagged = clean_inconclusive = 0
    transcode_total = transcode_caught = transcode_inconclusive = 0
    with tempfile.TemporaryDirectory(prefix="spectro-corpus-") as tmp:
        tmp = Path(tmp)
        for src in sources:
            src = Path(src)
            print(f"\n{src.name}")
            source_sr, verdict, lossy, cut, shelf = score(src)
            clean_total += 1
            flagged = verdict in ("WARN", "FAIL")
            clean_flagged += flagged
            clean_inconclusive += verdict == "INCONCLUSIVE"
            note = "   <- false positive" if flagged else "   <- inconclusive" if verdict == "INCONCLUSIVE" else ""
            print(f"  {'original':10} {verdict:13} {lossy:5.0f}  cutoff {cut:.0f} Hz ({shelf})"
                  f"{note}")

            # AAC and Vorbis can otherwise preserve a hi-res sample rate that
            # real distribution encodes rarely use. The measured corpus used
            # 48 kHz for sources above 48 kHz and native rate below it.
            rate_args = ["-ar", "48000"] if source_sr > 48000 else []

            for name, opts, ext in ENCODES:
                lossy_path = tmp / f"{src.stem}_{name}{ext}"
                back = tmp / f"{src.stem}_{name}.flac"
                if not ffmpeg(["-i", str(src)] + opts + rate_args + [str(lossy_path)]):
                    print(f"  {name:14} (encoder unavailable)")
                    continue
                if not ffmpeg(["-i", str(lossy_path), "-c:a", "flac", str(back)]):
                    continue
                _, verdict, lossy, cut, shelf = score(back)
                caught = verdict in ("FAIL", "WARN")
                transcode_total += 1
                transcode_caught += caught
                transcode_inconclusive += verdict == "INCONCLUSIVE"
                note = "" if caught else "   <- inconclusive" if verdict == "INCONCLUSIVE" else "   <- missed"
                print(f"  {name:14} {verdict:13} {lossy:5.0f}  cutoff {cut:.0f} Hz ({shelf}){note}")
                lossy_path.unlink(missing_ok=True)
                back.unlink(missing_ok=True)

    print(f"\nclean references: {clean_total - clean_flagged}/{clean_total} unflagged"
          f" ({clean_inconclusive} inconclusive)")
    print(f"transcodes: {transcode_caught}/{transcode_total} WARN/FAIL"
          f" ({transcode_inconclusive} inconclusive, "
          f"{transcode_total - transcode_caught - transcode_inconclusive} PASS)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
