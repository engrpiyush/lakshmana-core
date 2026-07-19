"""OIDC verification for push and scheduler deliveries (LLD §5, §13).

Pub/Sub and Cloud Scheduler both authenticate to the dispatcher with an OIDC token minted
for a service account we name in Terraform (`gatekeeper-push-sa`, the scheduler SA). The
service also runs on internal ingress, but that is a *network* control: it says the caller
is inside the perimeter, not that it is Pub/Sub. Anything inside the VPC could otherwise
POST a hand-written envelope and drive gates. So the token is verified as well.

Two deliberate positions:

* **The audience is optional, the identity is not.** A revision that has not been told its
  own URL still verifies the signature and the service account. An audience mismatch, when
  one *is* configured, is a hard failure — that is what stops a token minted for another
  service being replayed here.
* **An empty allow-list accepts any Google-issued identity.** It is the pre-Terraform
  default, and it is logged at WARNING on every request rather than silently trusted.

Failures return 401. Pub/Sub retries a 401 and the message eventually reaches the DLQ,
which is the right home for traffic we cannot authenticate.
"""

from __future__ import annotations

from dataclasses import dataclass

from gatekeeper.config import Config
from gatekeeper.logging import get_logger

__all__ = ["OidcVerifier", "TokenRejectedError", "verifier_for"]

log = get_logger(__name__)

_BEARER = "bearer "


class TokenRejectedError(Exception):
    """The caller could not be authenticated. Always a 401."""


@dataclass(slots=True)
class OidcVerifier:
    """Verifies the ``Authorization: Bearer`` header on an incoming request."""

    audience: str = ""
    allowed_service_accounts: frozenset[str] = frozenset()
    required: bool = True

    def verify(self, authorization: str | None) -> dict[str, object]:
        """Verify the header and return the token claims.

        Raises:
            TokenRejectedError: missing, malformed, unverifiable, or wrong-identity token.
        """
        if not self.required:
            log.warning("OIDC verification is disabled; accepting an unauthenticated request")
            return {}

        if not authorization or not authorization.lower().startswith(_BEARER):
            raise TokenRejectedError("missing or malformed Authorization header")

        token = authorization[len(_BEARER) :].strip()
        if not token:
            # `Bearer ` with nothing after it. The verifier below would reject it too, but
            # only by accident of how Google's parser handles an empty string; an empty
            # credential is refused here, deliberately, rather than delegated.
            raise TokenRejectedError("Authorization header carries an empty bearer token")

        claims = self._decode(token)

        email = str(claims.get("email") or "")
        if self.allowed_service_accounts and email not in self.allowed_service_accounts:
            raise TokenRejectedError(f"service account {email!r} may not call this service")
        if not self.allowed_service_accounts:
            log.warning(
                "no allowed service accounts configured; accepting any Google identity",
                fields={"email": email},
            )
        return claims

    def _decode(self, token: str) -> dict[str, object]:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        try:
            return dict(
                id_token.verify_oauth2_token(
                    token,
                    google_requests.Request(),
                    audience=self.audience or None,
                )
            )
        except (GoogleAuthError, ValueError) as exc:
            raise TokenRejectedError(f"token verification failed: {exc}") from exc


def verifier_for(config: Config) -> OidcVerifier:
    """Build the verifier this environment is configured for."""
    raw = config.get_str("gatekeeper.dispatcher.allowed-service-accounts")
    return OidcVerifier(
        audience=config.get_str("gatekeeper.dispatcher.oidc-audience"),
        allowed_service_accounts=frozenset(
            entry.strip() for entry in raw.split(",") if entry.strip()
        ),
        required=config.get_bool("gatekeeper.dispatcher.require-oidc"),
    )
