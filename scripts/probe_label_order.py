#!/usr/bin/env python3
"""Verify (or discover) each NLI artifact's output column order — VA-97 preflight.

An NLI head's three logits mean nothing until you know which column is contradiction, and
the roster disagrees with itself: ``modernbert-base-nli`` is entailment/neutral/
contradiction while ``nli-deberta-v3-small`` is contradiction/entailment/neutral. Get it
backwards and nothing crashes — G1 simply discards contradictions as NEUTRAL, quietly, on
every pair. That is the most expensive silent failure in the cascade, so the mapping is
measured rather than assumed.

The measurement is the 50-pair sanity fixture, which carries a ``lean`` of ENTAIL /
NEUTRAL / CONTRADICT per pair. For each candidate permutation of the three columns, score
the fixture and count how often the argmax lands on the labelled lean. The right
permutation wins by a wide margin; a near-tie means the fixture cannot settle it and the
artifact should not be swept until someone looks.

Usage::

    uv run python scripts/probe_label_order.py                 # every mirrored NLI row
    uv run python scripts/probe_label_order.py ettinx-nli-s@v1
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
SANITY_PAIRS = REPO_ROOT / "contracts" / "fixtures" / "model_sanity_pairs.jsonl"

CLASSES = ("entailment", "neutral", "contradiction")
LEAN_TO_CLASS = {
    "ENTAIL": "entailment",
    "NEUTRAL": "neutral",
    "CONTRADICT": "contradiction",
}

DECISIVE_MARGIN = 0.10
"""How far the best permutation must beat the runner-up to count as settled."""


def load_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def probe(ref: str, pairs: list[dict[str, str]]) -> tuple[tuple[str, ...], float, float]:
    """Return the winning column order, its accuracy, and the runner-up's."""
    from gatekeeper.config import load_config
    from gatekeeper.models.loader import loader_from_config
    from gatekeeper.models.roster import ModelTask
    from gatekeeper.scoring import EncoderScorer

    artifact = loader_from_config(load_config()).load(ref, expected_manifest_sha256="")

    # Score once through an identity mapping, which makes the returned fields *be* columns
    # 0/1/2; every candidate permutation is then a relabelling of the same numbers rather
    # than another few minutes of inference.
    scorer = EncoderScorer(artifact, ModelTask.NLI_3WAY, label_order=CLASSES)
    rows = scorer.score_nli([(pair["premise"], pair["hypothesis"]) for pair in pairs])
    scored = [[row.entailment, row.neutral, row.contradiction] for row in rows]

    truth = [LEAN_TO_CLASS[pair["lean"]] for pair in pairs]
    results: list[tuple[float, tuple[str, ...]]] = []
    for order in itertools.permutations(CLASSES):
        correct = sum(
            order[max(range(3), key=lambda column: row[column])] == expected
            for row, expected in zip(scored, truth, strict=True)
        )
        results.append((correct / len(pairs), order))

    results.sort(reverse=True)
    (best_accuracy, best_order), (runner_up, _) = results[0], results[1]
    return best_order, best_accuracy, runner_up


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refs", nargs="*", help="artifact refs; default is every NLI row")
    parser.add_argument("--local-dir", default="var/models")
    parser.add_argument("--cache-dir", default="var/model-cache")
    args = parser.parse_args(argv)

    os.environ.setdefault("GATEKEEPER_MODELS_LOCAL_DIR", args.local_dir)
    os.environ.setdefault("GATEKEEPER_MODELS_CACHE_DIR", args.cache_dir)

    from gatekeeper.models.roster import ALTERNATES, V1_ROSTER, ExportKind, ModelTask

    pairs = load_pairs(SANITY_PAIRS)
    candidates = [
        entry
        for entry in (*V1_ROSTER, *ALTERNATES)
        if entry.task is ModelTask.NLI_3WAY and entry.export is not ExportKind.CUSTOM
        if not args.refs or entry.ref in args.refs
    ]

    undecided = 0
    for entry in candidates:
        order, accuracy, runner_up = probe(entry.ref, pairs)
        margin = accuracy - runner_up
        verdict = "OK" if margin >= DECISIVE_MARGIN else "UNDECIDED"
        if verdict == "UNDECIDED":
            undecided += 1
        declared = entry.label_order or "(from id2label)"
        print(
            f"{verdict:9} {entry.ref:28} best={list(order)} "
            f"acc={accuracy:.2%} margin={margin:+.2%} declared={declared}"
        )

    if undecided:
        print(
            f"\n{undecided} artifact(s) the fixture cannot settle. Do not sweep them until "
            "the column order is established another way.",
            file=sys.stderr,
        )
    return 1 if undecided else 0


if __name__ == "__main__":
    sys.exit(main())
