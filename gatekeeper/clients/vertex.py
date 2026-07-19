"""The Vertex flash-lite door for G4 (LLD §8, §11, D-1).

D-1 puts the LLM tail in lakshmana, so this is the one place in the gatekeeper that spends
money. Everything here is shaped by that: calls are counted, tokens are counted, spend is
counted, and **the live door is off by default** — an unconfigured process gets
:class:`DryRunVertexClient` and cannot reach Vertex by accident. That mirrors
``gatekeeper.worker.execute-jobs``, and it is what lets the whole cascade run end to end on
a laptop while hard rule 4 (live calls are owner-gated, every time) still holds.

The wire shape is vishwamitra's ``VertexGeminiTransport``, deliberately: same ADC
credentials, same regional ``generateContent`` host, same ``thinkingConfig`` knob-selection
problem. That last one is not cosmetic — Gemini 3.x replaced the integer ``thinkingBudget``
with the enum ``thinkingLevel`` and **silently ignores the wrong knob**, so a pin bump from
a 2.5 row to a 3.x row would quietly restore full thinking and multiply the tail's bill.
:func:`thinking_config` ports ``GeminiThinking.config`` so the two services make the same
choice from the same model id.

**Retries are bounded and then given up on, per pair.** §11 makes G4's failure mode
explicit: quota/5xx beyond backoff is ``GK_E_VERTEX``, the affected *pairs* go to a human
with ``LLM_ERROR``, and the *gate still SUCCEEDS*. So the backoff lives here, exhaustion
raises :class:`~gatekeeper.errors.VertexError`, and the gate catches it one pair at a time
rather than failing the run — the partial-tail policy.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from gatekeeper.config import Config
from gatekeeper.errors import VertexError
from gatekeeper.logging import get_logger

__all__ = [
    "DryRunVertexClient",
    "VertexClient",
    "VertexGeminiClient",
    "VertexResponse",
    "client_for",
    "estimate_spend_usd",
    "thinking_config",
]

log = get_logger(__name__)

_RETRY_ATTEMPTS = 4
_BACKOFF_SECONDS = 0.5
_HTTP_OK = 200
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class VertexResponse:
    """One ``generateContent`` result: the text, and what it cost.

    Token counts come from the response's own ``usageMetadata`` rather than from a local
    tokenizer estimate — the bill is computed from Google's count, so the spend counter has
    to be too, or the "within 5% of actual billing" check has no chance.
    """

    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


class VertexClient(Protocol):
    """What G4 needs from a generative model: one prompt in, one answer out."""

    @property
    def model(self) -> str: ...

    @property
    def live(self) -> bool:
        """False for a double. G4 reports it on the run so a dry run is never mistaken."""
        ...

    def generate(self, prompt: str) -> VertexResponse: ...


def thinking_config(model: str, budget: int | None) -> dict[str, Any] | None:
    """The ``thinkingConfig`` for ``model``, or None to leave the model's default alone.

    Port of vishwamitra's ``GeminiThinking.config`` with the pin-directive branch dropped —
    lakshmana has one caller and no provider-row indirection, so the model id is the only
    signal. ``gemini-3*`` takes the enum (``0 → low``, anything else → ``high``); every
    older row takes the integer budget.
    """
    if budget is None:
        return None
    if model.startswith("gemini-3"):
        return {"thinkingLevel": "low" if budget == 0 else "high"}
    return {"thinkingBudget": budget}


def estimate_spend_usd(
    prompt_tokens: int,
    output_tokens: int,
    *,
    input_per_million: float,
    output_per_million: float,
) -> float:
    """Dollars for one call, from the response's own token counts and the configured rates.

    The rates are config, not constants: published prices move, and a number baked into
    source is a number nobody updates. A rate left at zero makes the counter report zero,
    which is visibly wrong rather than quietly wrong.
    """
    return (prompt_tokens / 1_000_000) * input_per_million + (
        output_tokens / 1_000_000
    ) * output_per_million


class DryRunVertexClient:
    """A double that never leaves the process (LLD §8; CLAUDE.md hard rule 4).

    It is not a test-only object: it is the **default** door, so a worker that nobody has
    explicitly pointed at Vertex runs the whole cascade — including G4's routing, cap and
    error paths — without spending anything. Every call is logged and counted, so a dry run
    still answers "how many pairs would the tail have cost me".

    The canned answer is deliberately a well-formed NEUTRAL with a rationale that *says* it
    is a dry run. A double that returned something a reader could mistake for a judgement
    is how a dry run ends up in an audit trail.
    """

    def __init__(self, model: str = "dry-run", *, answer: str | None = None) -> None:
        self._model = model
        self._answer = answer or (
            '[{"i":1,"relation":"NEUTRAL","confidence":0.5,'
            '"rationale":"Dry-run double: no model was called.","temporalNote":null}]'
        )
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def live(self) -> bool:
        return False

    def generate(self, prompt: str) -> VertexResponse:
        self.calls.append(prompt)
        log.info("dry-run G4 call (no Vertex request was made)", fields={"calls": len(self.calls)})
        # Token counts stay zero: inventing them would put a fictional number into
        # `llmSpendUsd`, and a dry run's honest cost is nothing.
        return VertexResponse(text=self._answer)


class VertexGeminiClient:
    """The live door — ADC + regional ``generateContent`` (vishwamitra's transport shape)."""

    def __init__(
        self,
        *,
        project_id: str,
        region: str,
        model: str,
        thinking_budget: int | None = None,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        session: Any = None,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._model = model
        self._thinking_budget = thinking_budget
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._session = session

    @property
    def model(self) -> str:
        return self._model

    @property
    def live(self) -> bool:
        return True

    @property
    def endpoint(self) -> str:
        # "global" reaches models not served regionally; anything else is in-region, which
        # is what O-4's `asia-southeast1` pin wants for data residency.
        host = (
            "aiplatform.googleapis.com"
            if self._region == "global"
            else f"{self._region}-aiplatform.googleapis.com"
        )
        return (
            f"https://{host}/v1/projects/{self._project_id}"
            f"/locations/{self._region}/publishers/google/models/{self._model}:generateContent"
        )

    def _authorized_session(self) -> Any:
        if self._session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._session = AuthorizedSession(credentials)
        return self._session

    def _body(self, prompt: str) -> dict[str, Any]:
        generation: dict[str, Any] = {
            "temperature": self._temperature,
            "maxOutputTokens": self._max_output_tokens,
            # The judge answers JSON, and asking for it in the response schema is cheaper
            # than paying for a retry when the model wraps it in prose.
            "responseMimeType": "application/json",
        }
        thinking = thinking_config(self._model, self._thinking_budget)
        if thinking is not None:
            generation["thinkingConfig"] = thinking
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }

    def generate(self, prompt: str) -> VertexResponse:
        """One call, with bounded jittered backoff on the retryable statuses.

        Raises:
            VertexError: once the budget is exhausted, or on a non-retryable status. §11
                turns that into ``LLM_ERROR`` on the affected pair — the gate survives.
        """
        session = self._authorized_session()
        last: str = ""

        for attempt in range(_RETRY_ATTEMPTS):
            try:
                response = session.post(self.endpoint, json=self._body(prompt), timeout=120)
            except Exception as exc:  # noqa: BLE001 — transport failures retry like a 503
                last = f"transport error: {exc}"
            else:
                if response.status_code == _HTTP_OK:
                    return _parse_generate_content(response.json())
                last = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise VertexError(f"vertex call failed and will not be retried — {last}")

            if attempt < _RETRY_ATTEMPTS - 1:
                delay = random.uniform(0, _BACKOFF_SECONDS * 2**attempt)
                log.warning(
                    "vertex call failed; backing off",
                    fields={"attempt": attempt + 1, "detail": last, "delaySeconds": delay},
                )
                time.sleep(delay)

        raise VertexError(f"vertex call failed after {_RETRY_ATTEMPTS} attempts — {last}")


def _parse_generate_content(payload: dict[str, Any]) -> VertexResponse:
    """Pull the text and the token counts out of a ``generateContent`` body.

    A 200 with no candidate is a **refusal**, not a success: safety blocks and recitation
    stops both land here. It raises, so the pair goes to a human with ``LLM_ERROR`` rather
    than being recorded as an empty verdict.
    """
    candidates = payload.get("candidates") or []
    parts = ((candidates[0] if candidates else {}).get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))

    usage = payload.get("usageMetadata") or {}
    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    # Thinking tokens are billed as output and reported separately; a tail that ignored
    # them would under-report its own bill by exactly the thinking budget.
    output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
        usage.get("thoughtsTokenCount") or 0
    )

    if not text.strip():
        reason = (candidates[0] if candidates else {}).get("finishReason") or payload.get(
            "promptFeedback"
        )
        raise VertexError(f"vertex returned no usable text (finishReason={reason!r})")

    return VertexResponse(text=text, prompt_tokens=prompt_tokens, output_tokens=output_tokens)


def client_for(config: Config, *, model: str, thinking_budget: int | None) -> VertexClient:
    """The door this process is allowed to use.

    ``model`` and ``thinking_budget`` come from the run's frozen ``configSnapshot.g4``, not
    from live config — the freeze invariant covers G4's model exactly as it covers the
    encoder gates' (LLD §8). Everything else here is *runtime* config: which project, which
    region, and whether live calls are permitted at all.

    Live calls require ``gatekeeper.gates.g4.live-calls`` to be set on. Off — the default —
    returns the dry-run double, so no deployment reaches Vertex without someone saying so.
    """
    if not config.get_bool("gatekeeper.gates.g4.live-calls"):
        log.warning(
            "G4 live calls are disabled; using the dry-run double",
            fields={"pinnedModel": model, "setting": "gatekeeper.gates.g4.live-calls"},
        )
        return DryRunVertexClient(model=model)

    return VertexGeminiClient(
        project_id=config.get_str("gatekeeper.firestore.project-id"),
        region=config.get_str("gatekeeper.gates.g4.region"),
        model=model,
        thinking_budget=thinking_budget,
        max_output_tokens=config.get_int("gatekeeper.gates.g4.max-output-tokens"),
    )
