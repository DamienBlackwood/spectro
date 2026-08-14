import json
import os
import time
from pathlib import Path

from ..audio import load_audio, looks_like_audio
from ..reporting import fmt_time
from ..term import RULE, paint, verdict_color, DIM

VERDICT_RANK = {"FAIL": 0, "WARN": 1, "INCONCLUSIVE": 2, "PASS": 3, "SKIP": 4, "ERROR": 5}


def analyze_file(path: str) -> dict:
    """Detect on one file, swallowing anything that goes wrong so one bad file
    doesn't take the whole run down."""
    from ..spectral import analyze_transcode_evidence

    p = Path(path)
    if not p.is_file() or not looks_like_audio(p):
        return {"file": path, "verdict": "SKIP", "note": "not audio"}
    try:
        data, sr = load_audio(str(p))
        res = analyze_transcode_evidence(data, sr)
        e = res.evidence
        return {
            "file": path, "verdict": e.verdict,
            "lossy_score": e.lossy_score, "quality_score": e.quality_score,
            "closest_resemblance": res.profile if e.hard_cutoff else None,
            "cutoff_hz": e.cutoff_freq, "shelf_type": e.shelf_type,
            "edge_p97_hz": e.edge_p97,
        }
    except (Exception, SystemExit) as ex:
        return {"file": path, "verdict": "ERROR", "note": str(ex)[:40]}


def _run(files, jobs: int):
    """Yield (index, report), in input order."""
    if jobs <= 1 or len(files) < 2:
        for i, fp in enumerate(files):
            yield i, analyze_file(fp)
        return

    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, report in enumerate(pool.map(analyze_file, files)):
            yield i, report


def cmd_batch(args, files, script_start: float) -> None:
    # the default is capped at 4, every worker holds a whole decoded track
    jobs = args.jobs if args.jobs else min(len(files), os.cpu_count() or 2, 4)
    print(f"\nBatch detect: {len(files)} files" + (f", {jobs} at a time" if jobs > 1 else "") + "\n")

    reports = []
    for i, report in _run(files, jobs):
        print(f"  [{i+1}/{len(files)}] {Path(report['file']).name}  → {report['verdict']}")
        reports.append(report)

    shown = [r for r in reports if not args.fails_only or r["verdict"] in ("FAIL", "WARN")]
    if args.sort:
        shown.sort(key=lambda r: (VERDICT_RANK.get(r["verdict"], 9), -r.get("lossy_score", 0)))

    if not shown:
        print("\n  No WARN or FAIL results to show.")
    else:
        rows = []
        for r in shown:
            name = Path(r["file"]).name
            if len(name) > 42:
                name = name[:39] + "..."
            rows.append((name, r["verdict"],
                         f"{r['lossy_score']:.0f}" if "lossy_score" in r else "",
                         f"{r['quality_score']:.0f}" if "quality_score" in r else "",
                         r.get("closest_resemblance") or r.get("note") or "-"))

        name_w = max(4, max(len(r[0]) for r in rows))
        print(f"\n  {'file':<{name_w}}  {'verdict':<13}{'lossy':>5}  {'quality':>7}  resembles")
        print("  " + RULE * (name_w + 40))
        for name, verdict, lossy, quality, profile in rows:
            color = verdict_color(verdict) if verdict in ("PASS", "WARN", "FAIL") else DIM
            v = paint(f"{verdict:<13}", color, bold=verdict in ("PASS", "WARN", "FAIL"))
            print(f"  {name:<{name_w}}  {v}{lossy:>5}  {quality:>7}  {profile}")

    if args.json is not None:
        json_path = args.json if args.json else "spectro_batch.json"
        with open(json_path, "w") as jf:
            json.dump(reports, jf, indent=2)
        print(f"\n  JSON report saved: {json_path}")

    total = time.perf_counter() - script_start
    print(f"\nDone in {fmt_time(total)}")
