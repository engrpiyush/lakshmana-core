"""Push and scheduler authentication (LLD §5, §13).

The dispatcher runs on internal ingress, but that is a network control and this is an
identity one. These tests pin the parts that are ours: the header contract, the service
account allow-list, and the fact that verification can be switched off only deliberately.
Google's own signature checking is not re-tested here — it is stubbed at ``_decode``, which
is the seam between our policy and their cryptography.
"""

from __future__ import annotations

import pytest

from gatekeeper.config import load_config
from gatekeeper.dispatcher.oidc import OidcVerifier, TokenRejectedError, verifier_for


class StubVerifier(OidcVerifier):
    """A verifier whose token decoding is canned, so the policy above it can be tested."""

    def __init__(self, claims, **kwargs):
        super().__init__(**kwargs)
        self._claims = claims

    def _decode(self, token):
        if token == "bad-token":
            raise TokenRejectedError("signature check failed")
        return self._claims


PUSH_SA = "gatekeeper-push-sa@example.iam.gserviceaccount.com"
OTHER_SA = "someone-else@example.iam.gserviceaccount.com"


# -- the header contract ------------------------------------------------------


@pytest.mark.parametrize("header", [None, "", "Token abc", "Bearer", "bearer "])
def test_a_missing_or_malformed_header_is_rejected(header) -> None:
    verifier = StubVerifier({"email": PUSH_SA})

    with pytest.raises(TokenRejectedError):
        verifier.verify(header)


def test_the_bearer_scheme_is_case_insensitive() -> None:
    """Pub/Sub sends ``Bearer``; nothing guarantees the casing survives a proxy."""
    verifier = StubVerifier({"email": PUSH_SA}, allowed_service_accounts=frozenset({PUSH_SA}))

    assert verifier.verify("bearer good-token")["email"] == PUSH_SA


def test_an_unverifiable_token_is_rejected() -> None:
    verifier = StubVerifier({"email": PUSH_SA})

    with pytest.raises(TokenRejectedError, match="signature"):
        verifier.verify("Bearer bad-token")


# -- the allow-list -----------------------------------------------------------


def test_an_allowed_service_account_is_accepted() -> None:
    verifier = StubVerifier({"email": PUSH_SA}, allowed_service_accounts=frozenset({PUSH_SA}))

    assert verifier.verify("Bearer good-token")["email"] == PUSH_SA


def test_a_service_account_outside_the_allow_list_is_rejected() -> None:
    """A valid Google token is not authorization — anyone can mint one."""
    verifier = StubVerifier({"email": OTHER_SA}, allowed_service_accounts=frozenset({PUSH_SA}))

    with pytest.raises(TokenRejectedError, match="may not call"):
        verifier.verify("Bearer good-token")


def test_an_empty_allow_list_accepts_any_google_identity() -> None:
    """The pre-Terraform default. It is permissive, and it warns on every request."""
    verifier = StubVerifier({"email": OTHER_SA})

    assert verifier.verify("Bearer good-token")["email"] == OTHER_SA


# -- switching it off ---------------------------------------------------------


def test_verification_can_be_disabled_for_local_runs() -> None:
    assert OidcVerifier(required=False).verify(None) == {}


def test_verification_is_required_by_default() -> None:
    """A deployment that forgets to configure OIDC must fail closed, not open."""
    with pytest.raises(TokenRejectedError):
        OidcVerifier().verify(None)


# -- config wiring ------------------------------------------------------------


def test_the_allow_list_is_parsed_from_a_comma_separated_setting() -> None:
    verifier = verifier_for(
        load_config(
            {"GATEKEEPER_DISPATCHER_ALLOWED_SERVICE_ACCOUNTS": f" {PUSH_SA} , {OTHER_SA} ,"}
        )
    )

    assert verifier.allowed_service_accounts == frozenset({PUSH_SA, OTHER_SA})


def test_oidc_is_on_by_default_in_config() -> None:
    assert verifier_for(load_config({})).required is True


def test_oidc_can_be_turned_off_by_env() -> None:
    assert (
        verifier_for(load_config({"GATEKEEPER_DISPATCHER_REQUIRE_OIDC": "false"})).required is False
    )
