"""The replay corpus exporter (VA-97 / LLD §15 step 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gatekeeper.replay.corpus import (
    CorpusStats,
    FieldMap,
    PairRecord,
    export_corpus,
    inspect_collection,
    read_corpus,
    write_corpus,
)


def edge(**overrides):
    document = {
        "claimAId": "claim-a",
        "claimBId": "claim-b",
        "claimAText": "Revenue grew 12% in Q3.",
        "claimBText": "Q3 revenue was up twelve percent.",
        "verdict": "CORROBORATES",
        "votes": ["CORROBORATES", "CORROBORATES", "NEUTRAL"],
        "golden": False,
        "method": "ENSEMBLE",
    }
    document.update(overrides)
    return document


def test_a_plain_edge_becomes_a_record() -> None:
    records = list(export_corpus([("pair-1", edge())]))

    assert len(records) == 1
    assert records[0].pair_id == "pair-1"
    assert records[0].ensemble_verdict == "CORROBORATES"
    assert records[0].votes == ("CORROBORATES", "CORROBORATES", "NEUTRAL")


def test_gatekeeper_authored_rows_are_never_labels() -> None:
    """The cascade's own prior output must not become its training signal (D-5)."""
    stats = CorpusStats()
    records = list(
        export_corpus(
            [("a", edge(method="GK_G1_NLI")), ("b", edge(method="ENSEMBLE"))], stats=stats
        )
    )

    assert [r.pair_id for r in records] == ["b"]
    assert stats.skipped_not_ensemble == 1


def test_vishwamitra_non_llm_methods_are_kept() -> None:
    records = list(export_corpus([("a", edge(method="RULE")), ("b", edge(method="EMBEDDING"))]))
    assert len(records) == 2


def test_a_pair_missing_text_is_skipped_and_counted() -> None:
    stats = CorpusStats()
    list(export_corpus([("a", edge(claimAText=""))], stats=stats))

    assert stats.skipped_unusable == 1
    assert stats.missing_fields["claimAText"] == 1


def test_a_pair_missing_a_verdict_is_skipped() -> None:
    stats = CorpusStats()
    list(export_corpus([("a", edge(verdict=None))], stats=stats))
    assert stats.skipped_unusable == 1


def test_alternate_field_names_resolve() -> None:
    """The verdict has been spelled more than one way; the first path that exists wins."""
    document = {
        "sourceClaimId": "a",
        "targetClaimId": "b",
        "textA": "one",
        "textB": "two",
        "ensembleVerdict": "neutral",
        "method": "ENSEMBLE",
    }
    records = list(export_corpus([("pair", document)]))

    assert records[0].ensemble_verdict == "NEUTRAL"
    assert records[0].claim_a_id == "a"


def test_an_explicit_field_map_overrides_the_defaults() -> None:
    document = {"aId": "a", "bId": "b", "ta": "one", "tb": "two", "judgement": "REPEATS"}
    field_map = FieldMap.from_json(
        {
            "claim_a_id": "aId",
            "claim_b_id": "bId",
            "claim_a_text": "ta",
            "claim_b_text": "tb",
            "verdict": ["judgement"],
        }
    )

    records = list(export_corpus([("pair", document)], field_map))

    assert records[0].ensemble_verdict == "REPEATS"


def test_a_field_map_refuses_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="not a corpus field"):
        FieldMap.from_json({"nonsense": "x"})


def test_nested_paths_are_followed() -> None:
    document = edge(votes=None)
    document["ensemble"] = {"votes": ["NEUTRAL", "NEUTRAL"]}
    records = list(export_corpus([("pair", document)]))
    assert records[0].votes == ("NEUTRAL", "NEUTRAL")


def test_votes_carried_as_records_are_flattened() -> None:
    document = edge(votes=[{"verdict": "NEUTRAL"}, {"verdict": "CORROBORATES"}])
    record = next(iter(export_corpus([("p", document)])))
    assert record.votes == ("NEUTRAL", "CORROBORATES")


def test_an_unrecognised_verdict_survives_verbatim() -> None:
    """Something unexpected in the label set is the owner's to see, not ours to drop."""
    records = list(export_corpus([("p", edge(verdict="PARTIALLY_SUPPORTS"))]))
    assert records[0].ensemble_verdict == "PARTIALLY_SUPPORTS"


def test_graph_text_wins_over_a_stale_snapshot() -> None:
    """The graph is the authority on claim text (LLD §8 hydrates from it at run time)."""
    stats = CorpusStats()
    records = list(
        export_corpus(
            [("p", edge(claimAText="", claimBText=""))],
            hydrate=lambda ids: {"claim-a": "fresh A", "claim-b": "fresh B"},
            stats=stats,
        )
    )

    assert records[0].claim_a_text == "fresh A"
    assert stats.hydrated_from_graph == 1


def test_hydration_is_not_attempted_when_text_is_already_present() -> None:
    calls: list[list[str]] = []

    def hydrate(ids):
        calls.append(list(ids))
        return {}

    list(export_corpus([("p", edge())], hydrate=hydrate))

    assert calls == []


def test_stats_summarise_the_export() -> None:
    stats = CorpusStats()
    list(
        export_corpus(
            [
                ("a", edge(verdict="NEUTRAL", golden=True)),
                ("b", edge(verdict="NEUTRAL", withContext=True)),
                ("c", edge(verdict="CONTRADICTS", golden=True)),
            ],
            stats=stats,
        )
    )

    assert stats.exported == 3
    assert stats.verdicts["NEUTRAL"] == 2
    assert stats.golden == 2
    assert stats.with_context == 1


# --- inspection ---------------------------------------------------------------------------


def test_inspect_reports_what_resolves_and_what_does_not() -> None:
    report = inspect_collection([{"claimAId": "a", "weirdVerdictName": "NEUTRAL"}])

    assert report["sampled"] == 1
    assert report["resolvedByDefaultMap"]["claim_a_id"] == "1/1"
    assert "verdict" in report["unresolved"]
    assert "weirdVerdictName" in report["topLevelKeys"]


def test_inspect_flattens_nested_keys() -> None:
    report = inspect_collection([{"ensemble": {"votes": []}}])
    assert "ensemble.votes" in report["topLevelKeys"]


# --- JSONL round trip -----------------------------------------------------------------------


def test_corpus_round_trips_through_jsonl(tmp_path: Path) -> None:
    records = list(
        export_corpus([("a", edge(golden=True, withContext=True, sourceSnippet="evidence"))])
    )
    path = tmp_path / "corpus.jsonl"

    assert write_corpus(records, path) == 1
    restored = read_corpus(path)

    assert restored == records
    assert restored[0].golden is True
    assert restored[0].source_snippet == "evidence"


def test_reading_a_file_that_is_not_a_corpus_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "other.jsonl"
    path.write_text('{"kind": "something-else"}\n')
    with pytest.raises(ValueError, match="not a replay corpus"):
        read_corpus(path)


def test_reading_a_future_schema_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"kind": "gatekeeperReplayCorpus", "schemaVersion": 99}\n')
    with pytest.raises(ValueError, match="schemaVersion"):
        read_corpus(path)


def test_an_empty_corpus_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("")
    assert read_corpus(path) == []


def test_a_record_needs_both_texts_and_a_verdict_to_be_usable() -> None:
    assert PairRecord("p", "a", "b", "one", "two", "NEUTRAL").usable
    assert not PairRecord("p", "a", "b", "", "two", "NEUTRAL").usable
    assert not PairRecord("p", "a", "b", "one", "two", "").usable
