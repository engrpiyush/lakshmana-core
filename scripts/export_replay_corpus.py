#!/usr/bin/env python3
"""Export vishwamitra's ensemble-labelled pairs to JSONL — VA-97 step 1 (LLD §15).

The source is a Firestore **backup restored into a separate emulator instance**. Never
point this at the shared dev emulator on 8082: that holds live development state, and a
restore into it would overwrite the owner's working data. Start your own::

    gcloud emulators firestore start --host-port=127.0.0.1:8092
    # restore the newest snapshot from vishwamitra-core/var/firestore-backups/ into it

Then, and this is the step worth not skipping::

    FIRESTORE_EMULATOR_HOST=127.0.0.1:8092 uv run python scripts/export_replay_corpus.py \\
        --project <project> --inspect

``--inspect`` reads a sample and prints which field names the collection actually uses and
which of the default mappings resolve against it. ``stage3_edges`` belongs to vishwamitra
and the LLD pins only the fields lakshmana *adds* to it, so the existing names — pair ids,
verdict, votes, golden markers — are worth confirming rather than assuming. Anything that
comes back unresolved goes in a field-map JSON::

    {"verdict": "judgeVerdict", "golden": ["humanConfirmed", "isGolden"]}

    ... --field-map var/replay/fieldmap.json --out var/replay/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gatekeeper.replay.corpus import (
    DEFAULT_COLLECTION,
    CorpusStats,
    FieldMap,
    export_corpus,
    inspect_collection,
    stream_edges,
    write_corpus,
)


def firestore_client(project: str):
    from gatekeeper.clients.firestore import firestore_client as gatekeeper_firestore_client
    from gatekeeper.config import load_config

    # Route through the gatekeeper client so gatekeeper.firestore.database
    # (GATEKEEPER_FIRESTORE_DATABASE) is honored — vishwamitra snapshots restore
    # into the named vishwakarma-labelling database, not (default).
    return gatekeeper_firestore_client(load_config(), project_id=project)


def graph_hydrator(config):
    """Claim id → text from Neo4j, batched. None when the graph is not configured."""
    from gatekeeper.clients.neo4j import Neo4jReader, neo4j_driver

    reader = Neo4jReader(neo4j_driver(config), database=config.get_str("gatekeeper.neo4j.database"))

    def hydrate(claim_ids):
        wanted = [claim_id for claim_id in claim_ids if claim_id]
        if not wanted:
            return {}
        rows = reader.run(
            "MATCH (c:Claim) WHERE c.claimId IN $ids RETURN c.claimId AS id, c.text AS text",
            ids=wanted,
        )
        return {row["id"]: row["text"] for row in rows if row.get("text")}

    return hydrate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Firestore project id in the emulator")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--stage3-run-id", default="", help="restrict to one Stage 3 run")
    parser.add_argument("--limit", type=int, default=0, help="cap documents scanned (0 = all)")
    parser.add_argument("--out", type=Path, default=Path("var/replay/corpus.jsonl"))
    parser.add_argument("--field-map", type=Path, default=None, help="JSON field-map overrides")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="report the collection's real field names and exit without writing",
    )
    parser.add_argument(
        "--hydrate-from-graph",
        action="store_true",
        help="fill missing claim text from Neo4j (the graph is the authority on it)",
    )
    args = parser.parse_args(argv)

    client = firestore_client(args.project)

    if args.inspect:
        documents = (document for _, document in stream_edges(client, args.collection))
        print(json.dumps(inspect_collection(documents), indent=2, sort_keys=True))
        return 0

    try:
        field_map = FieldMap.load(args.field_map)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    hydrate = None
    if args.hydrate_from_graph:
        from gatekeeper.config import load_config

        hydrate = graph_hydrator(load_config())

    stats = CorpusStats()
    written = write_corpus(
        export_corpus(
            stream_edges(client, args.collection, args.stage3_run_id, args.limit),
            field_map,
            hydrate=hydrate,
            stats=stats,
        ),
        args.out,
    )

    print(f"\n{stats.describe()}")
    print(f"\nwrote {written} pairs → {args.out}")

    if stats.missing_fields:
        print(
            "\nSome documents were unusable. The most likely cause is a field-map "
            "mismatch — rerun with --inspect and override the names that did not resolve:",
            file=sys.stderr,
        )
        for name, count in stats.missing_fields.most_common():
            print(f"  {name}: missing on {count} documents", file=sys.stderr)

    if written == 0:
        print("\nNothing exported; refusing to call that a success.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
