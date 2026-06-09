import json
import time
from pathlib import Path

from ..audio import load_audio, looks_like_audio
from ..reporting import fmt_time
from ..term import paint, verdict_color, DIM


def cmd_batch(args, files, script_start: float) -> None:
    from ..spectral import analyze_transcode_evidence

    print(f"\nBatch detect: {len(files)} files\n")
    rows = []
    reports = []

    for i, fp in enumerate(files, 1):
        p = Path(fp)
        name = p.name if len(p.name) <= 42 else p.name[:39] + "..."
        print(f"  [{i}/{len(files)}] {p.name}", end="", flush=True)

        if not p.is_file() or not looks_like_audio(p):
            rows.append((name, "SKIP", "", "", "not audio"))
            print("  → skipped")
            continue
        try:
            data, sr = load_audio(str(p))
            res = analyze_transcode_evidence(data, sr)
            e = res.evidence
            resembles = res.profile if (e.hard_cutoff or e.verdict != "PASS") else "-"
            rows.append((name, e.verdict, f"{e.lossy_score:.0f}", f"{e.quality_score:.0f}", resembles))
            reports.append({
                "file": str(p), "verdict": e.verdict,
                "lossy_score": e.lossy_score, "quality_score": e.quality_score,
                "closest_resemblance": res.profile,
                "cutoff_hz": e.cutoff_freq, "edge_p90_hz": e.edge_p90,
            })
            print(f"  → {e.verdict}")
        except (Exception, SystemExit) as ex:
            rows.append((name, "ERROR", "", "", str(ex)[:40]))
            print("  → error")

    name_w = max(4, max(len(r[0]) for r in rows))
    print(f"\n  {'file':<{name_w}}  {'verdict':<13}{'lossy':>5}  {'quality':>7}  resembles")
    print("  " + "─" * (name_w + 40))
    for name, verdict, lossy, quality, profile in rows:
        v = paint(f"{verdict:<13}", verdict_color(verdict), bold=True) if verdict in ("PASS", "WARN", "FAIL") else paint(f"{verdict:<13}", DIM)
        print(f"  {name:<{name_w}}  {v}{lossy:>5}  {quality:>7}  {profile}")

    if args.json is not None:
        json_path = args.json if args.json else "spectro_batch.json"
        with open(json_path, "w") as jf:
            json.dump(reports, jf, indent=2)
        print(f"\n  JSON report saved: {json_path}")

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
