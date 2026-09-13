"""Transactional transitions on ``gatekeeper_runs`` (LLD §6).

This module is the idempotency mechanism. Pub/Sub is at-least-once and unordered, so
every message is only an *attempted* transition: the Firestore transaction decides, and
losers become no-ops that are ACKed and dropped. ``runRequestId`` is the idempotency
root; the transaction is the enforcement (owner requirement #8).

Two invariants worth stating out loud, because both are load-bearing:

* **Reads precede writes.** Firestore transactions require it, and the supersede path
  reads a query before writing, so the ordering here is not stylistic.
* **A failure is not a crash.** The sweeper rescues expired leases; only an operator
  retrigger revives a ``FAILED`` gate (LLD §6 rule 3, owner requirement #7).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from google.api_core import exceptions as api_exceptions
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from gatekeeper.contracts.payload import GatekeeperRunRequest
from gatekeeper.enums import Gate, GateState, JudgeMode, RunMode, RunState, prior_gate
from gatekeeper.errors import ErrorCode, FirestoreTxnError
from gatekeeper.logging import get_logger
from gatekeeper.runs.model import COLLECTION, GatekeeperRun, RunTotals
from gatekeeper.timeutil import utc_now

__all__ = ["ClaimRejection", "ClaimResult", "GatekeeperRunStore", "SweepOutcome"]

log = get_logger(__name__)

DEFAULT_LEASE_MINUTES = 90

# Rounds of the whole transaction, each one a fresh transaction on top of the client's own
# internal retries, and the first backoff step they are spaced by (doubling, jittered).
_TXN_RETRY_ROUNDS = 4
_TXN_BACKOFF_SECONDS = 0.05

_TERMINAL_GATE_STATES = frozenset({GateState.SUCCEEDED, GateState.SKIPPED})


class SweepOutcome(StrEnum):
    """What a sweep did to one gate (LLD §11)."""

    RESCUED = "RESCUED"
    """Crashed work returned to PENDING for republishing."""

    STUCK = "STUCK"
    """Sweeps exhausted — gate and run FAILED with ``GK_E_STUCK``."""

    REPUBLISHED = "REPUBLISHED"
    """Nothing was wrong with the doc; the *message* went missing, so it is re-sent."""


class ClaimRejection(StrEnum):
    """Why a claim lost. Every one of these is an ACK-and-drop, logged as ``GK_DUP``."""

    RUN_NOT_FOUND = "RUN_NOT_FOUND"
    RUN_SUPERSEDED = "RUN_SUPERSEDED"
    RUN_TERMINAL = "RUN_TERMINAL"
    GATE_TERMINAL = "GATE_TERMINAL"
    GATE_LEASED = "GATE_LEASED"
    GATE_NOT_RETRIGGERABLE = "GATE_NOT_RETRIGGERABLE"
    PRIOR_GATE_NOT_SUCCEEDED = "PRIOR_GATE_NOT_SUCCEEDED"
    INVALID_ENTRY = "INVALID_ENTRY"


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """The outcome of a claim attempt. Only ``claimed`` may execute the worker job."""

    claimed: bool
    rejection: ClaimRejection | None = None
    run: GatekeeperRun | None = None
    superseded: tuple[str, ...] = ()


class GatekeeperRunStore:
    """Transactional access to ``gatekeeper_runs``."""

    def __init__(
        self,
        client: firestore.Client,
        *,
        collection: str = COLLECTION,
        lease_minutes: int = DEFAULT_LEASE_MINUTES,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._client = client
        self._collection = client.collection(collection)
        self._lease = timedelta(minutes=lease_minutes)
        self._clock = clock

    def _commit[T](self, transactional: Callable[..., T]) -> T:
        """Run a transactional callable, mapping contention onto ``GK_E_FIRESTORE_TXN``.

        The client library retries an aborted transaction a bounded number of times and
        then raises a bare ``ValueError``. That is exactly the ``GK_E_FIRESTORE_TXN``
        case from LLD §11, and callers need the code — the dispatcher turns it into a
        redelivery, the worker into a failed gate — so it must not surface as a generic
        error from somewhere inside Google's client.

        Those built-in retries carry no backoff, so contenders re-collide in lockstep and
        several dispatcher instances racing one message can *all* exhaust their budget,
        leaving the gate unclaimed until Pub/Sub redelivers. Safety never depended on
        this — the transaction still admits at most one winner — but liveness did, so
        each round gets a fresh transaction and a jittered pause to break the lockstep.
        """
        last: Exception | None = None
        for round_index in range(_TXN_RETRY_ROUNDS):
            try:
                return transactional(self._client.transaction())
            except (ValueError, api_exceptions.Aborted) as exc:
                last = exc
                if round_index < _TXN_RETRY_ROUNDS - 1:
                    time.sleep(random.uniform(0, _TXN_BACKOFF_SECONDS * 2**round_index))
        raise FirestoreTxnError(f"transaction contention was not resolved: {last}") from last

    # -- reads ----------------------------------------------------------------

    def get(self, run_request_id: str) -> GatekeeperRun | None:
        """Load a run doc, or None if it does not exist."""
        snapshot = self._collection.document(run_request_id).get()
        if not snapshot.exists:
            return None
        return GatekeeperRun.from_firestore(snapshot.to_dict())

    # -- claim ----------------------------------------------------------------

    def claim_gate(
        self,
        request: GatekeeperRunRequest,
        *,
        lease_owner: str,
        judge_mode: JudgeMode,
        config_snapshot: dict[str, Any],
    ) -> ClaimResult:
        """Transition ``request.gate`` from PENDING to RUNNING, or lose and no-op.

        The first claim of a FULL/G1 message creates the run doc, freezes
        ``configSnapshot``, and marks any non-terminal predecessor for the same
        ``stage3RunId`` SUPERSEDED — all in this one transaction (LLD §6 rules 1 and 4).

        Args:
            request: the parsed payload.
            lease_owner: identity recorded in the lease; the same value must be
                presented to commit or fail the gate.
            judge_mode: resolved from ``Stage3Run.paramsSnapshot`` (LLD §9).
            config_snapshot: frozen into a newly created run doc; ignored when the run
                already exists, because an in-flight run keeps its calibration.

        Returns:
            A :class:`ClaimResult`; ``claimed`` is False for every duplicate or stale
            message, which the caller should ACK and drop.
        """
        doc_ref = self._collection.document(request.run_request_id)
        now = self._clock()

        @firestore.transactional
        def _claim(transaction: firestore.Transaction) -> ClaimResult:
            snapshot = doc_ref.get(transaction=transaction)

            if not snapshot.exists:
                return self._create_and_claim(
                    transaction, doc_ref, request, lease_owner, judge_mode, config_snapshot, now
                )

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            rejection = self._reject_claim(run, request, now)
            if rejection is not None:
                return ClaimResult(False, rejection, run)

            transaction.update(doc_ref, self._claim_updates(request.gate, lease_owner, now, run))
            self._apply_claim_locally(run, request.gate, lease_owner, now)
            return ClaimResult(True, None, run)

        return self._commit(_claim)

    def _create_and_claim(
        self,
        transaction: firestore.Transaction,
        doc_ref: firestore.DocumentReference,
        request: GatekeeperRunRequest,
        lease_owner: str,
        judge_mode: JudgeMode,
        config_snapshot: dict[str, Any],
        now: datetime,
    ) -> ClaimResult:
        """Create the run doc on the first claim (LLD §14: never created by the client)."""
        # Only a FULL G1 message may open a run. A chained or FROM_GATE message for a
        # run that does not exist is stale by definition — its predecessor is gone.
        if request.gate is not Gate.G1_NEUTRAL or request.mode is not RunMode.FULL:
            return ClaimResult(False, ClaimRejection.INVALID_ENTRY, None)

        # READ before WRITE: the supersede scan must happen while we can still read.
        predecessors = self._non_terminal_predecessors(transaction, request)

        run = GatekeeperRun(
            schema_version=request.schema_version,
            run_request_id=request.run_request_id,
            intake_id=request.intake_id,
            stage3_run_id=request.stage3_run_id,
            request_timestamp=request.request_timestamp,
            triggered_by=request.triggered_by,
            judge_mode=judge_mode,
            mode=request.mode,
            state=RunState.RUNNING,
            config_snapshot=config_snapshot,
            totals=RunTotals(),
        )
        entry = run.gate(Gate.G1_NEUTRAL)
        entry.state = GateState.RUNNING
        entry.attempt = 1
        entry.lease_owner = lease_owner
        entry.lease_expires_at = now + self._lease
        entry.started_at = now

        document = run.to_firestore()
        document["createdAt"] = firestore.SERVER_TIMESTAMP
        document["updatedAt"] = firestore.SERVER_TIMESTAMP
        transaction.set(doc_ref, document)

        for predecessor in predecessors:
            transaction.update(
                predecessor.reference,
                {
                    "state": RunState.SUPERSEDED.value,
                    "supersededBy": request.run_request_id,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )

        return ClaimResult(
            True,
            None,
            run,
            tuple(doc.id for doc in predecessors),
        )

    def _non_terminal_predecessors(
        self, transaction: firestore.Transaction, request: GatekeeperRunRequest
    ) -> list[Any]:
        """Runs for the same ``stage3RunId`` that a FROM_START run must supersede.

        Only REQUESTED and RUNNING predecessors are superseded — the arrows the state
        machine actually draws (LLD §6, figure 3). A FAILED run keeps its state: it is
        the audit record of what went wrong, and it is already unreachable because
        reviving it needs a FROM_GATE message on its own runRequestId.

        Filtered on ``stage3RunId`` alone and narrowed in Python: the state predicate
        would otherwise demand a composite index for a query that returns a handful of
        docs (LLD §7.5 lists only the two indexes we actually need).
        """
        query = self._collection.where(
            filter=FieldFilter("stage3RunId", "==", request.stage3_run_id)
        )
        supersedable = {RunState.REQUESTED, RunState.RUNNING}
        predecessors = []
        for doc in query.get(transaction=transaction):
            if doc.id == request.run_request_id:
                continue
            state = (doc.to_dict() or {}).get("state")
            if state in {member.value for member in supersedable}:
                predecessors.append(doc)
        return predecessors

    def _reject_claim(
        self, run: GatekeeperRun, request: GatekeeperRunRequest, now: datetime
    ) -> ClaimRejection | None:
        """The precondition set from LLD §6 rule 1. None means the claim may proceed."""
        if run.state is RunState.SUPERSEDED:
            return ClaimRejection.RUN_SUPERSEDED
        if run.state is RunState.SUCCEEDED:
            return ClaimRejection.RUN_TERMINAL
        # A FAILED run is revived only by an operator retrigger, never by a redelivery.
        if run.state is RunState.FAILED and request.mode is not RunMode.FROM_GATE:
            return ClaimRejection.RUN_TERMINAL

        previous = prior_gate(request.gate)
        if previous is not None and run.gate(previous).state is not GateState.SUCCEEDED:
            return ClaimRejection.PRIOR_GATE_NOT_SUCCEEDED

        entry = run.gate(request.gate)
        if entry.lease_held_at(now):
            return ClaimRejection.GATE_LEASED
        if entry.state in _TERMINAL_GATE_STATES:
            return ClaimRejection.GATE_TERMINAL

        if request.mode is RunMode.FROM_GATE:
            # FROM_GATE is valid only on a FAILED gate or a lease-expired RUNNING one.
            if entry.state not in {GateState.FAILED, GateState.RUNNING}:
                return ClaimRejection.GATE_NOT_RETRIGGERABLE
        elif entry.state is GateState.FAILED:
            # A chained FULL message cannot restart failed work; the operator must.
            return ClaimRejection.GATE_TERMINAL

        return None

    def _claim_updates(
        self, gate: Gate, lease_owner: str, now: datetime, run: GatekeeperRun
    ) -> dict[str, Any]:
        prefix = f"gates.{gate.value}"
        return {
            f"{prefix}.state": GateState.RUNNING.value,
            f"{prefix}.attempt": run.gate(gate).attempt + 1,
            f"{prefix}.leaseOwner": lease_owner,
            f"{prefix}.leaseExpiresAt": now + self._lease,
            f"{prefix}.startedAt": now,
            f"{prefix}.endedAt": None,
            f"{prefix}.errorCode": None,
            f"{prefix}.errorDetail": None,
            "state": RunState.RUNNING.value,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

    def _apply_claim_locally(
        self, run: GatekeeperRun, gate: Gate, lease_owner: str, now: datetime
    ) -> None:
        """Mirror the claim onto the in-memory run so ``ClaimResult.run`` is post-claim.

        Callers read ``attempt`` off the result for logging; handing them the stale
        pre-transaction value would misreport every retry by one.
        """
        entry = run.gate(gate)
        entry.state = GateState.RUNNING
        entry.attempt += 1
        entry.lease_owner = lease_owner
        entry.lease_expires_at = now + self._lease
        entry.started_at = now
        entry.ended_at = None
        entry.error_code = None
        entry.error_detail = None
        run.state = RunState.RUNNING

    # -- gate outcomes --------------------------------------------------------

    def commit_gate(
        self,
        run_request_id: str,
        gate: Gate,
        *,
        lease_owner: str,
        counters: dict[str, Any] | None = None,
    ) -> bool:
        """Mark ``gate`` SUCCEEDED with its counters.

        Always called **before** publishing the next gate: the crash window between
        commit and publish is what the sweeper rescues, and the reverse order would
        double-run a gate instead (LLD §5).

        Returns:
            True if the commit applied; False if the lease was lost or the gate had
            already moved on — in which case this worker must not publish.
        """
        return self._settle_gate(
            run_request_id,
            gate,
            lease_owner=lease_owner,
            gate_state=GateState.SUCCEEDED,
            counters=counters,
        )

    def fail_gate(
        self,
        run_request_id: str,
        gate: Gate,
        *,
        lease_owner: str,
        error_code: ErrorCode,
        error_detail: str,
        counters: dict[str, Any] | None = None,
    ) -> bool:
        """Mark ``gate`` FAILED and the run FAILED. No automatic retry follows."""
        return self._settle_gate(
            run_request_id,
            gate,
            lease_owner=lease_owner,
            gate_state=GateState.FAILED,
            counters=counters,
            error_code=error_code,
            error_detail=error_detail,
        )

    def _settle_gate(
        self,
        run_request_id: str,
        gate: Gate,
        *,
        lease_owner: str,
        gate_state: GateState,
        counters: dict[str, Any] | None,
        error_code: ErrorCode | None = None,
        error_detail: str | None = None,
    ) -> bool:
        doc_ref = self._collection.document(run_request_id)
        now = self._clock()
        prefix = f"gates.{gate.value}"

        @firestore.transactional
        def _settle(transaction: firestore.Transaction) -> bool:
            snapshot = doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            if run.state is RunState.SUPERSEDED:
                return False

            entry = run.gate(gate)
            # Only the lease holder settles the gate: a worker whose lease expired and
            # was swept must not overwrite the rescuer's work.
            if entry.state is not GateState.RUNNING or entry.lease_owner != lease_owner:
                return False

            updates: dict[str, Any] = {
                f"{prefix}.state": gate_state.value,
                f"{prefix}.endedAt": now,
                f"{prefix}.leaseOwner": None,
                f"{prefix}.leaseExpiresAt": None,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
            if counters is not None:
                updates[f"{prefix}.counters"] = dict(counters)
            if gate_state is GateState.FAILED:
                updates[f"{prefix}.errorCode"] = error_code.value if error_code else None
                updates[f"{prefix}.errorDetail"] = error_detail
                updates["state"] = RunState.FAILED.value

            transaction.update(doc_ref, updates)
            return True

        return self._commit(_settle)

    def renew_gate_lease(self, run_request_id: str, gate: Gate, *, lease_owner: str) -> bool:
        """Extend the gate lease for the worker that holds it — the heartbeat (LLD §11, B2).

        The gate lease was written once at claim and never renewed, so a gate that outlived
        its 90-minute lease — a 40k-pair G1 pass does by 6-7x — looked dead to the sweeper
        while it was still running, and a second worker was started on the same gate. A live
        worker calling this once per batch keeps ``leaseExpiresAt`` (and ``updatedAt``, which
        the sweeper's stalled-run pass reads) ahead of the sweeper, so only a genuinely dead
        worker is ever rescued.

        It doubles as the loss detector. If the lease has already moved — a rescue happened
        while this worker was stalled — it returns False, and the caller must stop without
        writing rather than race the rescuer for the same pairs (B3). The same is true once
        the run itself is no longer live: a SUPERSEDED run's straggler learns to stop here.

        This is the fourth and only other write site for the gate lease, alongside claim,
        settle-clear and rescue-clear; like all of them it is a single transaction so at most
        one outcome is ever observed.

        Returns:
            True if the caller still holds a RUNNING lease on this gate (now extended);
            False if the run is gone, terminal or superseded, or the lease has moved on.
        """
        doc_ref = self._collection.document(run_request_id)
        now = self._clock()
        prefix = f"gates.{gate.value}"

        @firestore.transactional
        def _renew(transaction: firestore.Transaction) -> bool:
            snapshot = doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            if run.state not in {RunState.REQUESTED, RunState.RUNNING}:
                return False

            entry = run.gate(gate)
            # Only the current holder renews. A worker whose lease was swept finds either a
            # different owner or a gate no longer RUNNING here, and takes that as its cue to
            # abort — exactly the check `_settle_gate` makes before committing.
            if entry.state is not GateState.RUNNING or entry.lease_owner != lease_owner:
                return False

            transaction.update(
                doc_ref,
                {
                    f"{prefix}.leaseExpiresAt": now + self._lease,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            return True

        return self._commit(_renew)

    # -- sweep ----------------------------------------------------------------

    def active_runs(self) -> list[GatekeeperRun]:
        """Every run the sweeper might have to act on.

        Filtered on ``state`` alone and never on the gate map: Firestore cannot query
        inside ``gates.*`` without an index per gate, and the set of non-terminal runs is
        small by construction (one per in-flight intake). Narrowing in Python is the same
        trade ``_non_terminal_predecessors`` makes.
        """
        runs: list[GatekeeperRun] = []
        for state in (RunState.REQUESTED, RunState.RUNNING):
            query = self._collection.where(filter=FieldFilter("state", "==", state.value))
            runs.extend(GatekeeperRun.from_firestore(doc.to_dict()) for doc in query.get())
        return runs

    def rescue_gate(
        self, run_request_id: str, gate: Gate, *, max_sweeps: int
    ) -> SweepOutcome | None:
        """Return a crashed gate to PENDING, or fail the run once sweeps are exhausted.

        The precondition is re-checked **inside** the transaction against the same lease
        the caller saw expire. That is what makes a rescue happen exactly once: two
        sweepers racing the same gate both read an expired lease, and the loser finds the
        gate already PENDING (or leased by the winner's republish) and does nothing.

        Only crashes are rescued. A ``FAILED`` gate is not touched here at all — it waits
        for an operator retrigger (LLD §6 rule 3, owner requirement #7).

        Returns:
            The outcome, or None if another sweeper got there first.
        """
        doc_ref = self._collection.document(run_request_id)
        now = self._clock()
        prefix = f"gates.{gate.value}"

        @firestore.transactional
        def _rescue(transaction: firestore.Transaction) -> SweepOutcome | None:
            snapshot = doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            if run.state not in {RunState.REQUESTED, RunState.RUNNING}:
                return None

            entry = run.gate(gate)
            if entry.state is not GateState.RUNNING or entry.lease_held_at(now):
                # Either already rescued, or the worker is alive after all.
                return None

            if entry.sweep_attempt >= max_sweeps:
                transaction.update(
                    doc_ref,
                    {
                        f"{prefix}.state": GateState.FAILED.value,
                        f"{prefix}.endedAt": now,
                        f"{prefix}.leaseOwner": None,
                        f"{prefix}.leaseExpiresAt": None,
                        f"{prefix}.errorCode": ErrorCode.GK_E_STUCK.value,
                        f"{prefix}.errorDetail": (
                            f"lease expired {entry.sweep_attempt} times; sweeps exhausted"
                        ),
                        "state": RunState.FAILED.value,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    },
                )
                return SweepOutcome.STUCK

            transaction.update(
                doc_ref,
                {
                    f"{prefix}.state": GateState.PENDING.value,
                    f"{prefix}.sweepAttempt": entry.sweep_attempt + 1,
                    f"{prefix}.leaseOwner": None,
                    f"{prefix}.leaseExpiresAt": None,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            return SweepOutcome.RESCUED

        return self._commit(_rescue)

    # -- finalize -------------------------------------------------------------

    def fail_run(self, run_request_id: str, *, error_code: ErrorCode, error_detail: str) -> bool:
        """Fail a run without touching any gate — FINALIZE's failure path (LLD §8).

        Every other failure in the system is a *gate* failure, and :meth:`fail_gate` is how
        it is recorded. FINALIZE has no gate: it runs after G4 has legitimately committed,
        so there is nothing to mark FAILED and pushing G4 back to FAILED would be a lie
        about which step broke — and would make a FROM_GATE retrigger re-run the whole
        expensive tail to fix a bookkeeping problem.

        The code and detail land on the run doc's own ``errorCode``/``errorDetail``, which is
        where the runbook's "which gate, what code" lookup falls back to when no gate entry
        carries one.

        Returns:
            True if the run moved to FAILED; False if it was already terminal.
        """
        doc_ref = self._collection.document(run_request_id)
        now = self._clock()

        @firestore.transactional
        def _fail(transaction: firestore.Transaction) -> bool:
            snapshot = doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            if run.state is not RunState.RUNNING:
                return False

            transaction.update(
                doc_ref,
                {
                    "state": RunState.FAILED.value,
                    "errorCode": error_code.value,
                    "errorDetail": error_detail,
                    "endedAt": now,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            return True

        return self._commit(_fail)

    def finalize_run(self, run_request_id: str, totals: RunTotals) -> bool:
        """FINALIZE: write ``totals`` and set the run SUCCEEDED (LLD §8).

        Publishes nothing — vishwamitra's existing poll observes the state change and
        advances its own lifecycle (LLD §7.4, owner requirement #6).

        Returns:
            True if the run was finalized; False if it was not RUNNING or any gate is
            still unsettled.
        """
        doc_ref = self._collection.document(run_request_id)

        @firestore.transactional
        def _finalize(transaction: firestore.Transaction) -> bool:
            snapshot = doc_ref.get(transaction=transaction)
            if not snapshot.exists:
                return False

            run = GatekeeperRun.from_firestore(snapshot.to_dict())
            if run.state is not RunState.RUNNING:
                return False
            if any(
                entry.state not in (_TERMINAL_GATE_STATES | {GateState.FAILED})
                for entry in run.gates.values()
            ):
                return False
            if any(entry.state is GateState.FAILED for entry in run.gates.values()):
                return False

            transaction.update(
                doc_ref,
                {
                    "state": RunState.SUCCEEDED.value,
                    "totals": totals.to_firestore(),
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )
            return True

        return self._commit(_finalize)
