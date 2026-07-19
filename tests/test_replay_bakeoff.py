"""Bake-off metrics, the cascade replay, and the GO bar (VA-97 / LLD §15).

These numbers decide a go/no-go, so they are tested against hand-built cases where the
right answer is countable by hand rather than only ever observed on a corpus.
"""

from __future__ import annotations

from pathlib import Path

from gatekeeper.enums import Verdict
from gatekeeper.gates.decisions import Disposition, Thresholds
from gatekeeper.replay.bakeoff import (
    PairScores,
    RosterCandidate,
    ScoreCache,
    best_cell,
    best_passing,
    build_report,
    corpus_fingerprint,
    grid_search,
    replay,
    run_cascade,
    score_corpus,
    scorer_factory,
    sweep,
)
from gatekeeper.replay.corpus import PairRecord
from gatekeeper.replay.metrics import (
    GoBar,
    GoBarResult,
    Outcome,
    RosterReport,
    contradicts_recall,
    coverage_curve,
    evaluate_go_bar,
    expected_calibration_error,
    per_verdict_scores,
    volumes,
)
from gatekeeper.scoring.encoder import GroundingScore, NliScores

T = Thresholds()


def outcome(
    pair_id: str,
    disposition: Disposition,
    verdict: Verdict | None,
    ensemble: str,
    *,
    golden: bool = False,
    confidence: float | None = None,
) -> Outcome:
    return Outcome(
        pair_id=pair_id,
        disposition=disposition,
        gate="G1_NEUTRAL",
        verdict=verdict,
        ensemble_verdict=ensemble,
        golden=golden,
        confidence=confidence,
    )


DECIDED = Disposition.DECIDED
G4 = Disposition.ESCALATE_G4
HUMAN = Disposition.ROUTE_HUMAN


# --- per-verdict P/R -----------------------------------------------------------------------


def test_precision_and_recall_are_counted_per_verdict() -> None:
    outcomes = [
        outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL"),
        outcome("2", DECIDED, Verdict.NEUTRAL, "NEUTRAL"),
        outcome("3", DECIDED, Verdict.NEUTRAL, "CORROBORATES"),  # a precision miss
        outcome("4", G4, None, "NEUTRAL"),  # a recall miss, not a precision one
    ]

    neutral = next(s for s in per_verdict_scores(outcomes) if s.label == "NEUTRAL")

    assert neutral.predicted == 3
    assert neutral.actual == 3
    assert neutral.true_positives == 2
    assert neutral.precision == 2 / 3
    assert neutral.recall == 2 / 3


def test_escalating_never_costs_precision() -> None:
    """A deferred pair has not been judged wrongly — that is the whole trade."""
    decided_only = [outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL")]
    with_escalations = decided_only + [
        outcome(str(i), G4, None, "CORROBORATES") for i in range(2, 10)
    ]

    before = next(s for s in per_verdict_scores(decided_only) if s.label == "NEUTRAL")
    after = next(s for s in per_verdict_scores(with_escalations) if s.label == "NEUTRAL")

    assert before.precision == after.precision == 1.0


def test_a_verdict_nobody_predicted_scores_zero_not_an_error() -> None:
    scores = per_verdict_scores([outcome("1", G4, None, "REPEATS")])
    repeats = next(s for s in scores if s.label == "REPEATS")
    assert repeats.precision == 0.0 and repeats.recall == 0.0


# --- NEUTRAL coverage curve ----------------------------------------------------------------


def test_coverage_is_measured_over_the_whole_corpus() -> None:
    """Not over the pairs G1 happened to look at — that would flatter the gate."""
    outcomes = [
        outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL"),
        outcome("2", DECIDED, Verdict.NEUTRAL, "NEUTRAL"),
        outcome("3", G4, None, "CORROBORATES"),
        outcome("4", G4, None, "CORROBORATES"),
    ]

    point = coverage_curve([(0.95, outcomes)])[0]

    assert point["coverage"] == 0.5
    assert point["precision"] == 1.0
    assert point["g4Share"] == 0.5


def test_the_curve_has_one_point_per_swept_threshold() -> None:
    outcomes = [outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL")]
    curve = coverage_curve([(0.80, outcomes), (0.95, outcomes)])
    assert [point["value"] for point in curve] == [0.80, 0.95]


# --- CONTRADICTS recall --------------------------------------------------------------------


def test_a_golden_contradiction_that_reaches_a_human_is_preserved() -> None:
    outcomes = [outcome("1", HUMAN, None, "CONTRADICTS", golden=True)]
    assert contradicts_recall(outcomes)["recall"] == 1.0


def test_a_golden_contradiction_sent_to_g4_is_also_preserved() -> None:
    """Deferring to the LLM tail is not discarding it."""
    outcomes = [outcome("1", G4, None, "CONTRADICTS", golden=True)]
    assert contradicts_recall(outcomes)["recall"] == 1.0


def test_a_golden_contradiction_decided_as_neutral_is_a_discard() -> None:
    """The failure the escape hatch exists to prevent."""
    outcomes = [
        outcome("1", HUMAN, None, "CONTRADICTS", golden=True),
        outcome("2", DECIDED, Verdict.NEUTRAL, "CONTRADICTS", golden=True),
    ]

    result = contradicts_recall(outcomes)

    assert result["recall"] == 0.5
    assert result["discarded"] == 1.0


def test_non_golden_contradictions_are_not_the_denominator() -> None:
    outcomes = [
        outcome("1", DECIDED, Verdict.NEUTRAL, "CONTRADICTS", golden=False),
        outcome("2", HUMAN, None, "CONTRADICTS", golden=True),
    ]
    assert contradicts_recall(outcomes)["goldenContradictions"] == 1.0


def test_a_corpus_with_no_golden_contradictions_reports_zero_not_a_crash() -> None:
    assert contradicts_recall([outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL")])["recall"] == 0.0


# --- calibration ---------------------------------------------------------------------------


def test_a_perfectly_calibrated_set_has_near_zero_ece() -> None:
    outcomes = [
        outcome(str(i), DECIDED, Verdict.CORROBORATES, "CORROBORATES", confidence=0.95)
        for i in range(19)
    ] + [outcome("miss", DECIDED, Verdict.CORROBORATES, "NEUTRAL", confidence=0.95)]

    assert expected_calibration_error(outcomes)["ece"] < 0.01


def test_overconfidence_shows_up_as_ece() -> None:
    outcomes = [
        outcome(str(i), DECIDED, Verdict.CORROBORATES, "NEUTRAL", confidence=0.99)
        for i in range(10)
    ]
    assert expected_calibration_error(outcomes)["ece"] > 0.9


def test_pairs_without_a_confidence_are_not_calibration_samples() -> None:
    assert expected_calibration_error([outcome("1", G4, None, "NEUTRAL")])["samples"] == 0.0


# --- funnel --------------------------------------------------------------------------------


def test_the_funnel_splits_the_corpus() -> None:
    outcomes = [
        outcome("1", DECIDED, Verdict.NEUTRAL, "NEUTRAL"),
        outcome("2", G4, None, "NEUTRAL"),
        outcome("3", HUMAN, None, "CONTRADICTS"),
        outcome("4", DECIDED, Verdict.REPEATS, "REPEATS"),
    ]

    funnel = volumes(outcomes)

    assert funnel["decidedByEncoders"] == 0.5
    assert funnel["escalatedToG4"] == 0.25
    assert funnel["routedToHuman"] == 0.25


# --- the GO bar ----------------------------------------------------------------------------


def _passing_corpus() -> list[Outcome]:
    """70% neutral at perfect precision, one preserved golden contradiction, no G4."""
    outcomes = [outcome(f"n{i}", DECIDED, Verdict.NEUTRAL, "NEUTRAL") for i in range(70)]
    outcomes += [outcome(f"c{i}", HUMAN, None, "CONTRADICTS", golden=True) for i in range(10)]
    outcomes += [outcome(f"g{i}", DECIDED, Verdict.CORROBORATES, "CORROBORATES") for i in range(20)]
    return outcomes


def test_a_clean_corpus_passes_the_go_bar() -> None:
    assert evaluate_go_bar(_passing_corpus()).passed


def test_thin_neutral_coverage_fails() -> None:
    outcomes = [outcome(f"n{i}", DECIDED, Verdict.NEUTRAL, "NEUTRAL") for i in range(50)]
    outcomes += [outcome(f"g{i}", DECIDED, Verdict.CORROBORATES, "CORROBORATES") for i in range(50)]
    outcomes += [outcome("c", HUMAN, None, "CONTRADICTS", golden=True)]

    result = evaluate_go_bar(outcomes)

    assert not result.passed
    assert not result.criteria["neutralCoverage"]["passed"]
    assert result.criteria["neutralPrecision"]["passed"]


def test_a_discarded_golden_contradiction_fails_the_bar() -> None:
    outcomes = _passing_corpus()
    outcomes[70] = outcome("c0", DECIDED, Verdict.NEUTRAL, "CONTRADICTS", golden=True)

    result = evaluate_go_bar(outcomes)

    assert not result.passed
    assert not result.criteria["contradictsRecall"]["passed"]


def test_too_much_g4_fails_the_bar() -> None:
    outcomes = _passing_corpus() + [outcome(f"e{i}", G4, None, "NEUTRAL") for i in range(20)]

    result = evaluate_go_bar(outcomes)

    assert not result.passed
    assert not result.criteria["g4Share"]["passed"]


def _corpus_without_contradictions() -> list[Outcome]:
    """Session03's real shape: no golden pairs, no CONTRADICTS verdict anywhere."""
    outcomes = [outcome(f"n{i}", DECIDED, Verdict.NEUTRAL, "NEUTRAL") for i in range(70)]
    outcomes += [outcome(f"g{i}", DECIDED, Verdict.CORROBORATES, "CORROBORATES") for i in range(30)]
    return outcomes


def test_an_unmeasurable_criterion_is_neither_passed_nor_failed() -> None:
    result = evaluate_go_bar(_corpus_without_contradictions())

    assert result.criteria["contradictsRecall"]["measurable"] is False
    assert result.unmeasurable == ("contradictsRecall",)


def test_a_corpus_that_cannot_test_contradictions_still_clears_what_it_can() -> None:
    """The other three criteria are real measurements and must not be held hostage."""
    result = evaluate_go_bar(_corpus_without_contradictions())

    assert result.passed
    assert result.criteria["neutralPrecision"]["passed"]


def test_an_untested_criterion_never_reads_as_a_met_one() -> None:
    """The failure mode that matters: a vacuous pass looking like evidence."""
    described = evaluate_go_bar(_corpus_without_contradictions()).describe()

    assert "PARTIAL" in described
    assert "n/a  contradictsRecall" in described
    assert "ok  contradictsRecall" not in described


def test_a_measurable_contradiction_criterion_is_still_enforced() -> None:
    """Marking the empty case unmeasurable must not disarm the populated one."""
    outcomes = _passing_corpus()
    outcomes[70] = outcome("c0", DECIDED, Verdict.NEUTRAL, "CONTRADICTS", golden=True)

    result = evaluate_go_bar(outcomes)

    assert result.criteria["contradictsRecall"]["measurable"] is True
    assert not result.passed
    assert result.unmeasurable == ()


def test_the_bar_numbers_are_the_lld_ones() -> None:
    bar = GoBar()
    assert bar.min_neutral_precision == 0.95
    assert bar.min_neutral_coverage == 0.60
    assert bar.min_contradicts_recall_ratio == 0.95
    assert bar.max_g4_share == 0.05


# --- cascade replay ------------------------------------------------------------------------


def nli(entailment: float, neutral: float, contradiction: float) -> NliScores:
    return NliScores(entailment=entailment, neutral=neutral, contradiction=contradiction)


def pair_scores(
    *,
    g1_contradiction: float = 0.0,
    g1_neutral: float = 0.98,
    support: float = 0.10,
    g3_contradiction: float = 0.0,
) -> PairScores:
    from gatekeeper.gates.decisions import G1Scores, G2Scores, G3Scores

    entailment = max(0.0, 1.0 - g1_neutral - g1_contradiction)
    return PairScores(
        g1=G1Scores(
            forward=nli(entailment, g1_neutral, g1_contradiction),
            backward=nli(entailment, g1_neutral, g1_contradiction),
        ),
        g2=G2Scores(forward=GroundingScore(support), backward=GroundingScore(support)),
        g3=G3Scores(
            family_b_forward=nli(0.0, 1 - g3_contradiction, g3_contradiction),
            family_b_backward=nli(0.0, 1 - g3_contradiction, g3_contradiction),
        ),
    )


def record(pair_id: str = "p", **overrides) -> PairRecord:
    fields = {
        "pair_id": pair_id,
        "claim_a_id": "a",
        "claim_b_id": "b",
        "claim_a_text": "one",
        "claim_b_text": "two",
        "ensemble_verdict": "NEUTRAL",
    }
    fields.update(overrides)
    return PairRecord(**fields)


def test_a_quiet_pair_stops_at_g1() -> None:
    result = run_cascade(record(), pair_scores(), T)
    assert result.gate == "G1_NEUTRAL"
    assert result.verdict is Verdict.NEUTRAL


def test_a_supported_pair_reaches_g2() -> None:
    scores = pair_scores(g1_neutral=0.50, support=0.95)
    result = run_cascade(record(), scores, T)
    assert result.gate == "G2_CORROBORATION"
    assert result.verdict is Verdict.CORROBORATES


def test_an_undecided_pair_reaches_g3() -> None:
    result = run_cascade(record(), pair_scores(g1_neutral=0.50, support=0.10), T)
    assert result.gate == "G3_CONTRADICTION"


def test_a_contradiction_flag_reaches_g3_even_when_g2_would_corroborate() -> None:
    """§8 puts contradiction-flagged pairs in front of G3 whatever else happens to them."""
    scores = pair_scores(g1_contradiction=0.50, g1_neutral=0.40, support=0.99)

    result = run_cascade(record(), scores, T)

    assert result.gate == "G3_CONTRADICTION"
    assert result.contradiction_flag is True
    assert result.verdict is not Verdict.CORROBORATES


def test_a_two_family_contradiction_ends_with_a_human() -> None:
    scores = pair_scores(g1_contradiction=0.95, g1_neutral=0.04, g3_contradiction=0.92)
    result = run_cascade(record(ensemble_verdict="CONTRADICTS", golden=True), scores, T)
    assert result.disposition is Disposition.ROUTE_HUMAN


def test_replay_covers_every_pair_it_has_scores_for() -> None:
    records = [record("a"), record("b"), record("c")]
    scores = {r.pair_id: pair_scores() for r in records}
    assert len(replay(records, scores, T)) == 3


def test_replay_skips_pairs_with_no_scores() -> None:
    records = [record("a"), record("b")]
    assert len(replay(records, {"a": pair_scores()}, T)) == 1


# --- sweeping ------------------------------------------------------------------------------


def test_raising_neutral_min_lowers_coverage() -> None:
    """The trade the curve exists to show, on scores that sit between the two thresholds.

    The second family disagrees here, so G3 cannot rescue what G1 stops deciding — see
    the next test for what happens when it can.
    """
    records = [record(f"p{i}") for i in range(10)]
    scores = {r.pair_id: pair_scores(g1_neutral=0.93, g3_contradiction=0.50) for r in records}

    curve = coverage_curve(sweep(records, scores, T, "neutral_min", [0.90, 0.95]))

    assert curve[0]["coverage"] == 1.0
    assert curve[1]["coverage"] == 0.0


def test_g3_can_still_call_neutral_on_a_pair_g1_stopped_deciding() -> None:
    """NEUTRAL coverage is not controlled by `neutralMin` alone.

    A pair both families agree is quiet gets decided at G3 by two-family consensus even
    once G1's bar is above it. Anyone reading the coverage curve as "the G1 threshold
    curve" would misread the cost of tightening it.
    """
    records = [record("p")]
    scores = {"p": pair_scores(g1_neutral=0.93, g3_contradiction=0.0)}

    tightened = replay(records, scores, Thresholds(neutral_min=0.95))

    assert tightened[0].gate == "G3_CONTRADICTION"
    assert tightened[0].verdict is Verdict.NEUTRAL


def test_a_sweep_does_not_mutate_the_base_thresholds() -> None:
    records = [record()]
    scores = {"p": pair_scores()}
    sweep(records, scores, T, "neutral_min", [0.1, 0.9])
    assert T.neutral_min == 0.95


# --- grid search ---------------------------------------------------------------------------


def test_the_grid_covers_every_combination() -> None:
    records = [record()]
    scores = {"p": pair_scores()}

    cells = grid_search(
        records, scores, T, {"neutralMin": [0.8, 0.9], "contraEscape": [0.02, 0.2, 0.5]}
    )

    assert len(cells) == 6
    assert {(c.values["neutralMin"], c.values["contraEscape"]) for c in cells} == {
        (0.8, 0.02),
        (0.8, 0.2),
        (0.8, 0.5),
        (0.9, 0.02),
        (0.9, 0.2),
        (0.9, 0.5),
    }


def test_a_grid_cell_carries_the_thresholds_that_produced_it() -> None:
    cells = grid_search([record()], {"p": pair_scores()}, T, {"contraEscape": [0.42]})

    assert cells[0].thresholds.contra_escape == 0.42
    assert cells[0].thresholds.neutral_min == T.neutral_min  # untouched axes survive


def test_the_grid_finds_an_operating_point_one_axis_sweeps_cannot() -> None:
    """The escape hatch gates `neutralMin`: at 0.02 nothing reaches the neutral test.

    This is why the go/no-go uses --grid. Sweeping `neutralMin` alone with the escape
    hatch at its proposed value returns a flat curve that says nothing.
    """
    records = [record(f"p{i}") for i in range(10)]
    # Quiet pairs carrying enough contradiction signal to trip a 0.02 escape hatch, and
    # enough that G3's two-family consensus will not quietly rescue them either.
    scores = {r.pair_id: pair_scores(g1_neutral=0.85, g1_contradiction=0.15) for r in records}

    flat = coverage_curve(sweep(records, scores, T, "neutral_min", [0.80, 0.90, 0.95]))
    assert {point["coverage"] for point in flat} == {0.0}

    cells = grid_search(records, scores, T, {"neutralMin": [0.80], "contraEscape": [0.02, 0.20]})
    by_escape = {cell.values["contraEscape"]: cell.neutral["coverage"] for cell in cells}

    assert by_escape[0.02] == 0.0
    assert by_escape[0.20] == 1.0


def test_best_cell_is_none_when_nothing_passes() -> None:
    records = [record("c", ensemble_verdict="CONTRADICTS", golden=True)]
    scores = {"c": pair_scores(g1_neutral=0.99)}

    cells = grid_search(records, scores, T, {"neutralMin": [0.90, 0.95]})

    assert best_cell(cells) is None


def test_a_grid_report_adopts_the_best_passing_thresholds() -> None:
    """§15's bar, applied mechanically — the report is at the cell that passed."""
    records = [record(f"n{i}") for i in range(19)]
    records += [record("c", ensemble_verdict="CONTRADICTS", golden=True)]
    scores = {r.pair_id: pair_scores(g1_neutral=0.85, g1_contradiction=0.15) for r in records[:19]}
    scores["c"] = pair_scores(g1_contradiction=0.95, g1_neutral=0.04, g3_contradiction=0.90)

    report = build_report(
        CANDIDATE,
        records,
        scores,
        Thresholds(neutral_min=0.80),
        grid={"contraEscape": [0.02, 0.20]},
    )

    assert report.go_bar.passed
    assert report.thresholds["contraEscape"] == 0.20
    assert len(report.grid) == 2


def test_a_grid_that_finds_nothing_still_reports_what_was_measured() -> None:
    records = [record("c", ensemble_verdict="CONTRADICTS", golden=True)]
    scores = {"c": pair_scores(g1_neutral=0.99)}

    report = build_report(CANDIDATE, records, scores, T, grid={"neutralMin": [0.90]})

    assert not report.go_bar.passed
    assert report.thresholds["neutralMin"] == T.neutral_min
    assert report.pairs == 1


# --- score cache ---------------------------------------------------------------------------


CANDIDATE = RosterCandidate(name="v1", g1="a@v1", g2="b@v1", g3="c@v1")


class CountingScorer:
    """A scorer that answers constantly and counts how much inference was asked of it."""

    def __init__(self, calls: dict[str, int], ref: str) -> None:
        self.calls = calls
        self.ref = ref

    def score_nli(self, pairs: list[tuple[str, str]]) -> list[NliScores]:
        self.calls[self.ref] = self.calls.get(self.ref, 0) + len(pairs)
        return [nli(0.2, 0.7, 0.1) for _ in pairs]

    def score_grounding(self, pairs: list[tuple[str, str]]) -> list[GroundingScore]:
        self.calls[self.ref] = self.calls.get(self.ref, 0) + len(pairs)
        return [GroundingScore(support=0.3) for _ in pairs]


def counting_factory(calls: dict[str, int]):
    return lambda ref: CountingScorer(calls, ref)


def test_scores_round_trip_through_the_cache(tmp_path: Path) -> None:
    records = [record(f"n{i}") for i in range(3)]
    cache = ScoreCache(tmp_path, corpus_fingerprint(records))

    first = score_corpus(records, CANDIDATE, counting_factory({}), cache=cache)
    second = score_corpus(records, CANDIDATE, counting_factory({}), cache=cache)

    assert second.keys() == first.keys()
    assert second["n0"].g1.forward.neutral == first["n0"].g1.forward.neutral
    assert second["n0"].g2.forward.support == first["n0"].g2.forward.support


def test_a_second_roster_reuses_the_artifacts_it_shares(tmp_path: Path) -> None:
    """The saving the artifact-keyed cache exists for: shared weights are scored once."""
    records = [record(f"n{i}") for i in range(3)]
    cache = ScoreCache(tmp_path, corpus_fingerprint(records))
    calls: dict[str, int] = {}

    score_corpus(records, CANDIDATE, counting_factory(calls), cache=cache)
    first_pass = dict(calls)
    # Same G2 and G3, a different G1 — only the new artifact should cost anything.
    score_corpus(
        records,
        RosterCandidate(name="alt", g1="z@v1", g2="b@v1", g3="c@v1"),
        counting_factory(calls),
        cache=cache,
    )

    assert calls["b@v1"] == first_pass["b@v1"]
    assert calls["c@v1"] == first_pass["c@v1"]
    assert calls["z@v1"] > 0


def test_a_cache_over_a_different_corpus_is_refused(tmp_path: Path) -> None:
    """A --limit smoke run must not leave a cache a full run silently reports from."""
    subset = [record(f"n{i}") for i in range(2)]
    full = [record(f"n{i}") for i in range(5)]
    calls: dict[str, int] = {}

    subset_cache = ScoreCache(tmp_path, corpus_fingerprint(subset))
    full_cache = ScoreCache(tmp_path, corpus_fingerprint(full))

    score_corpus(subset, CANDIDATE, counting_factory(calls), cache=subset_cache)
    scores = score_corpus(full, CANDIDATE, counting_factory(calls), cache=full_cache)

    assert len(scores) == 5


def test_an_absent_cache_is_a_miss_not_an_error(tmp_path: Path) -> None:
    records = [record("n0")]
    cache = ScoreCache(tmp_path / "nope", corpus_fingerprint(records))

    assert len(score_corpus(records, CANDIDATE, counting_factory({}), cache=cache)) == 1


def test_the_context_variant_survives_the_cache(tmp_path: Path) -> None:
    records = [
        PairRecord(
            pair_id="n0",
            claim_a_id="a",
            claim_a_text="a",
            claim_b_id="b",
            claim_b_text="b",
            context_a="because a",
            context_b="because b",
            with_context=True,
            ensemble_verdict="NEUTRAL",
        )
    ]
    cache = ScoreCache(tmp_path, corpus_fingerprint(records))

    score_corpus(records, CANDIDATE, counting_factory({}), cache=cache)
    restored = score_corpus(records, CANDIDATE, counting_factory({}), cache=cache)

    assert restored["n0"].g1.with_context
    assert restored["n0"].g1.context_forward is not None


# --- report --------------------------------------------------------------------------------


def test_a_report_carries_everything_section_15_asks_for() -> None:
    records = [record(f"n{i}") for i in range(8)]
    records += [record("c", ensemble_verdict="CONTRADICTS", golden=True)]
    scores = {r.pair_id: pair_scores() for r in records[:8]}
    scores["c"] = pair_scores(g1_contradiction=0.95, g1_neutral=0.04, g3_contradiction=0.90)

    report = build_report(
        CANDIDATE,
        records,
        scores,
        T,
        sweeps={"neutralMin": [0.90, 0.95], "contraEscape": [0.02, 0.30]},
    )

    document = report.to_json()
    assert document["pairs"] == 9
    assert len(document["sweeps"]["neutralMin"]) == 2
    assert len(document["sweeps"]["contraEscape"]) == 2
    assert {"perVerdict", "contradictions", "calibration", "funnel", "goBar"} <= document.keys()
    assert document["goBar"]["passed"] is True


def test_best_passing_returns_nothing_when_no_roster_passes() -> None:
    records = [record("c", ensemble_verdict="CONTRADICTS", golden=True)]
    scores = {"c": pair_scores(g1_neutral=0.99)}  # discards the golden contradiction
    report = build_report(CANDIDATE, records, scores, T)

    assert not report.go_bar.passed
    assert best_passing([report]) is None


def _report_with(name: str, coverage: float, g4_share: float) -> RosterReport:
    """A report stub — this is about the ranking rule, not about re-deriving metrics."""
    return RosterReport(
        roster=name,
        thresholds={},
        artifacts={},
        per_verdict=[],
        neutral={"coverage": coverage, "precision": 1.0, "decided": 0.0},
        sweeps={},
        contradictions={"recall": 1.0},
        calibration={"ece": 0.0},
        funnel={"escalatedToG4": g4_share},
        go_bar=GoBarResult(passed=True),
    )


def test_best_passing_prefers_the_higher_neutral_coverage() -> None:
    """The cascade exists to remove judging load, so coverage ranks first."""
    stingy = _report_with("stingy", coverage=0.62, g4_share=0.01)
    generous = _report_with("generous", coverage=0.85, g4_share=0.04)

    assert best_passing([stingy, generous]) is generous


def test_best_passing_breaks_a_coverage_tie_on_g4_cost() -> None:
    cheap = _report_with("cheap", coverage=0.80, g4_share=0.01)
    dear = _report_with("dear", coverage=0.80, g4_share=0.04)

    assert best_passing([dear, cheap]) is cheap


def test_best_passing_ignores_a_failing_roster_however_good_its_coverage() -> None:
    failing = _report_with("failing", coverage=0.99, g4_share=0.0)
    failing.go_bar = GoBarResult(passed=False)
    passing = _report_with("passing", coverage=0.61, g4_share=0.0)

    assert best_passing([failing, passing]) is passing


# --- scorer lifetime -----------------------------------------------------------------------


class RecordingLoader:
    """Counts artifact loads so scorer reuse and release are both observable."""

    def __init__(self) -> None:
        self.loads: list[str] = []

    def load(self, ref: str, expected_manifest_sha256: str = "") -> object:
        self.loads.append(ref)
        return ref


def test_a_scorer_is_reused_while_it_is_the_live_one(monkeypatch) -> None:
    loader = RecordingLoader()
    monkeypatch.setattr("gatekeeper.replay.bakeoff.scorer_for_artifact", lambda a, **k: a)
    build = scorer_factory(loader)

    build("a@v1")
    build("a@v1")

    assert loader.loads == ["a@v1"]


def test_only_max_live_scorers_are_held_at_once(monkeypatch) -> None:
    """At fp32 the roster totals ~5.6 GB; holding every artifact would exhaust the box."""
    loader = RecordingLoader()
    monkeypatch.setattr("gatekeeper.replay.bakeoff.scorer_for_artifact", lambda a, **k: a)
    build = scorer_factory(loader)

    build("a@v1")
    build("b@v1")
    build("a@v1")

    # a@v1 was released to make room for b@v1, so coming back to it costs a reload.
    assert loader.loads == ["a@v1", "b@v1", "a@v1"]
