#!/usr/bin/env python3
"""The replay bake-off — VA-97, the go/no-go (LLD §15).

One command produces the full report for a named roster::

    uv run python scripts/replay_bakeoff.py --corpus var/replay/corpus.jsonl \\
        --all-rosters \\
        --grid contraEscape=0.02,0.05,0.1,0.2,0.3,0.5 \\
        --grid neutralMin=0.80,0.85,0.90,0.95 \\
        --grid neutralConsensus=0.05,0.1,0.2,0.4 \\
        --report var/replay/report.json

Zero LLM spend: every gate in the cascade is an encoder, and G4 is only ever *counted*
here, never called. Artifacts come from the local mirror through the production loader, so
a bad digest fails the same way it would in a deployed worker.

Inference is cached per roster (``--score-cache-dir``). Threshold search replays the cached
probabilities through the real decision functions, so a few hundred combinations cost
seconds against the hours of inference behind them.

Prefer ``--grid`` over ``--sweep`` for the go/no-go. The thresholds gate each other — with
``contraEscape`` at its proposed 0.02 the escape hatch flags most pairs before ``neutralMin``
is ever consulted — so one-axis sweeps come back flat and say nothing about where the
operating point is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROSTERS: dict[str, tuple[str, str, str]] = {
    # name: (G1, G2, G3) — LLD §8's v1 row, then the alternates worth a sweep.
    "v1": ("modernbert-base-nli@v1", "minicheck-deberta-l@v1", "deberta-mnli-fever-anli@v1"),
    "v1-factcg": ("modernbert-base-nli@v1", "factcg-deberta-l@v1", "deberta-mnli-fever-anli@v1"),
    "v1-vitaminc": ("modernbert-base-nli@v1", "minicheck-deberta-l@v1", "vitaminc-mnli@v1"),
    "cheap": ("nli-deberta-v3-small@v1", "minicheck-deberta-l@v1", "deberta-mnli-fever-anli@v1"),
    "ettinx": ("ettinx-nli-s@v1", "minicheck-deberta-l@v1", "deberta-mnli-fever-anli@v1"),
    "ettinx-vitaminc": ("ettinx-nli-s@v1", "minicheck-deberta-l@v1", "vitaminc-mnli@v1"),
}


def parse_axis(raw: str) -> tuple[str, tuple[float, ...]]:
    """``contraEscape=0.02,0.1,0.3`` → ``("contraEscape", (0.02, 0.1, 0.3))``."""
    axis, _, values = raw.partition("=")
    if not values:
        raise ValueError(f"{raw!r} is not AXIS=v1,v2,...")
    return axis.strip(), tuple(float(value) for value in values.split(",") if value.strip())


def resolve_thresholds(config, args, sweepable, thresholds_cls):
    """Base thresholds from config plus ``--set``, and the ``--sweep`` axes.

    Raises:
        ValueError: on a malformed or unknown axis. A typo'd threshold name must not
            silently produce a report of the defaults.
    """
    sweeps = dict(parse_axis(raw) for raw in args.sweep)
    grid = dict(parse_axis(raw) for raw in args.grid)
    overrides = {axis: values[0] for axis, values in map(parse_axis, args.overrides)}

    unknown = [axis for axis in (*sweeps, *grid, *overrides) if axis not in sweepable]
    if unknown:
        raise ValueError(f"unknown threshold(s) {unknown}; known: {sorted(sweepable)}")

    thresholds = thresholds_cls.from_snapshot(config.snapshot())
    if overrides:
        thresholds = replace(
            thresholds, **{sweepable[axis]: value for axis, value in overrides.items()}
        )
        print(f"base thresholds overridden: {overrides}")
    return thresholds, sweeps, grid


def render(report) -> str:
    lines = [
        f"\n=== {report.roster} ===",
        f"  artifacts: {report.artifacts}",
        f"  pairs: {report.pairs} · wall clock {report.wall_clock_seconds:.1f}s",
        f"  thresholds: {report.thresholds}",
        "  per-verdict:",
    ]
    for score in report.per_verdict:
        lines.append(
            f"    {score.label:14} P={score.precision:.4f} R={score.recall:.4f} "
            f"F1={score.f1:.4f}  (predicted {score.predicted}, actual {score.actual})"
        )
    funnel = report.funnel
    lines += [
        f"  NEUTRAL: precision {report.neutral['precision']:.4f} "
        f"at {report.neutral['coverage']:.2%} coverage",
        f"  funnel: encoders {funnel['decidedByEncoders']:.2%} · "
        f"G4 {funnel['escalatedToG4']:.2%} · human {funnel['routedToHuman']:.2%}",
        f"  CONTRADICTS recall on golden: {report.contradictions['recall']:.4f} "
        f"({int(report.contradictions['preserved'])}/"
        f"{int(report.contradictions['goldenContradictions'])} preserved, "
        f"{int(report.contradictions['discarded'])} discarded)",
        f"  calibration: ECE {report.calibration['ece']:.4f} "
        f"over {int(report.calibration['samples'])} scored decisions",
    ]
    for axis, curve in report.sweeps.items():
        lines.append(f"  sweep {axis}:")
        for point in curve:
            lines.append(
                f"    {axis}={point['value']:<7.3f} NEUTRAL coverage={point['coverage']:7.2%} "
                f"precision={point['precision']:.4f}  G4={point['g4Share']:6.2%} "
                f"human={point['humanShare']:6.2%}"
            )
    if report.grid:
        passing = [cell for cell in report.grid if cell["passed"]]
        lines.append(f"  grid: {len(passing)}/{len(report.grid)} combinations passed the bar")
    lines.append("  GO bar: " + report.go_bar.describe().replace("\n", "\n  "))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--roster",
        action="append",
        default=[],
        help=f"roster name, repeatable. Known: {', '.join(sorted(ROSTERS))}",
    )
    parser.add_argument("--all-rosters", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N pairs")
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="AXIS=v1,v2",
        help="sweep one threshold, repeatable — e.g. --sweep contraEscape=0.02,0.1,0.3. "
        "The escape hatch usually moves coverage and G4 volume far more than neutralMin.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="AXIS=v1,v2",
        help="joint grid search over threshold combinations, repeatable. Unlike --sweep "
        "these are searched together, which is the only way to find an operating point "
        "when the thresholds gate each other. The best passing cell becomes the report's "
        "thresholds — this is §15's bar applied mechanically.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="AXIS=value",
        help="move a base threshold before reporting, repeatable",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report", type=Path, default=Path("var/replay/report.json"))
    parser.add_argument("--score-cache-dir", type=Path, default=Path("var/replay/scores"))
    parser.add_argument("--local-dir", default="var/models")
    parser.add_argument("--cache-dir", default="var/model-cache")
    args = parser.parse_args(argv)

    os.environ.setdefault("GATEKEEPER_MODELS_LOCAL_DIR", args.local_dir)
    os.environ.setdefault("GATEKEEPER_MODELS_CACHE_DIR", args.cache_dir)

    from gatekeeper.config import load_config
    from gatekeeper.gates.decisions import Thresholds
    from gatekeeper.models.loader import loader_from_config
    from gatekeeper.replay.bakeoff import (
        SWEEPABLE,
        RosterCandidate,
        ScoreCache,
        best_passing,
        build_report,
        corpus_fingerprint,
        score_corpus,
        scorer_factory,
    )
    from gatekeeper.replay.corpus import read_corpus

    names = sorted(ROSTERS) if args.all_rosters else (args.roster or ["v1"])
    unknown = [name for name in names if name not in ROSTERS]
    if unknown:
        print(f"error: unknown roster(s) {unknown}; known: {sorted(ROSTERS)}", file=sys.stderr)
        return 2

    records = read_corpus(args.corpus)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"error: {args.corpus} has no pairs", file=sys.stderr)
        return 2

    config = load_config()
    loader = loader_from_config(config)
    scorer_for = scorer_factory(loader, batch_size=args.batch_size)

    try:
        thresholds, sweeps, grid = resolve_thresholds(config, args, SWEEPABLE, Thresholds)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    fingerprint = corpus_fingerprint(records)
    print(f"corpus: {len(records)} pairs from {args.corpus} (fingerprint {fingerprint})")
    # One cache for the whole table: it is keyed by artifact, so the rosters' shared models
    # are scored once between them rather than once each.
    cache = ScoreCache(args.score_cache_dir, fingerprint)

    reports = []
    for name in names:
        g1, g2, g3 = ROSTERS[name]
        candidate = RosterCandidate(name=name, g1=g1, g2=g2, g3=g3)

        started = time.monotonic()
        scores = score_corpus(
            records,
            candidate,
            scorer_for,
            cache=cache,
            progress=lambda slot, _done, total: print(f"  scoring {slot} over {total} pairs…"),
        )
        elapsed = time.monotonic() - started

        report = build_report(
            candidate,
            records,
            scores,
            thresholds,
            sweeps=sweeps,
            grid=grid,
            wall_clock_seconds=elapsed,
            precisions={
                ref: str(loader.load(ref, expected_manifest_sha256="").precision)
                for ref in candidate.refs
            },
        )
        reports.append(report)
        print(render(report))

    return archive(reports, records, args, best_passing(reports))


def archive(reports, records, args, winner) -> int:
    """Write the report the owner reviews asynchronously, and return the exit code."""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "pairs": len(records),
                "adopted": winner.roster if winner else None,
                "reports": [report.to_json() for report in reports],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nreport → {args.report}")

    if winner is None:
        # Session03's park condition: the architecture is unchanged, but no combination
        # of the mirrored rosters clears §15's bar, and that is the owner's call.
        print(
            "\nNO-GO: no roster passed the GO bar. Nothing gate-shaped should be built "
            "on these numbers.",
            file=sys.stderr,
        )
        return 1

    print(f"\nGO: adopt roster '{winner.roster}' — {winner.artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
