"""
app/auth/jwt.py
───────────────
JWT token validation + tenant extraction (A6a of the Lane A blueprint).

Supports HS256 (shared secret, simple) and RS256 (asymmetric, public
key verification). The choice is driven by ``VALONY_JWT_ALGORITHM``;
both have legitimate use cases:

  * HS256 — fine for a single-tenant deploy where the API issues its
    own tokens. Fast, no key infrastructure.
  * RS256 — required when tokens come from an external IdP (Auth0,
    Cognito, Okta). The IdP signs with a private key, the API
    verifies with the IdP's published public key (JWKS).

Validation order:
  1. Token present and well-formed (Bearer header parses).
  2. Signature verifies under the configured key/secret.
  3. ``exp`` is in the future (with configurable leeway).
  4. ``iss`` matches expected issuer (when configured).
  5. ``aud`` matches expected audience (when configured).

Successful validation returns a ``TokenClaims`` with first-class
``tenant_id`` / ``user_id`` / ``roles`` extracted from common claim
names plus the raw payload for any caller that needs deeper fields.

Why first-class ``tenant_id``: A6b will scope every persistence write
and every memory read by tenant. Forcing it through a typed field at
the auth boundary catches missing-tenant tokens at the door rather
than discovering them deep in a pgvector query.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────
class AuthError(Exception):
    """Base for every auth-related rejection."""


class MissingToken(AuthError):
    """No bearer token in the request, or empty after the prefix."""


class InvalidToken(AuthError):
    """Token is malformed, signature doesn't verify, issuer/audience
    mismatch, or any other validation failure that isn't expiry."""


class ExpiredToken(AuthError):
    """``exp`` claim is in the past (after leeway)."""


# ──────────────────────────────────────────────────────────────
# Config + claims
# ──────────────────────────────────────────────────────────────
@dataclass
class JWTConfig:
    """Decoded JWT validation parameters. Build via
    ``resolve_jwt_config()`` for the standard env-driven config or
    instantiate directly in tests."""

    algorithm: str = "HS256"            # 'HS256' or 'RS256'
    secret: str | None = None           # for HS256
    public_key: str | None = None       # for RS256 (PEM-encoded)
    issuer: str | None = None           # validate iss when set
    audience: str | None = None         # validate aud when set
    leeway_seconds: int = 30            # clock-skew tolerance
    tenant_claim: str = "tenant_id"     # which claim names the tenant
    user_claim: str = "sub"             # which claim names the user
    roles_claim: str = "roles"          # which claim lists roles


@dataclass
class TokenClaims:
    """Validated, first-class projection of a JWT payload."""

    tenant_id: str
    user_id: str | None = None
    roles: list[str] = field(default_factory=list)
    expires_at: int | None = None       # UNIX ts from `exp`
    raw: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────
def decode_token(token: str, config: JWTConfig) -> TokenClaims:
    """Validate ``token`` and return its claims.

    Raises ``MissingToken``, ``InvalidToken``, or ``ExpiredToken``
    on failure. Never returns partial / unverified data — caller can
    trust ``tenant_id`` after this returns.
    """
    if not token:
        raise MissingToken("token is empty")

    # Lazy import so the rest of the auth surface (errors, dataclasses)
    # is importable without PyJWT installed — useful for tests that
    # only exercise the env-resolution + middleware-disabled paths.
    try:
        import jwt as pyjwt
        from jwt.exceptions import (
            ExpiredSignatureError,
            InvalidAudienceError,
            InvalidIssuerError,
            InvalidSignatureError,
            InvalidTokenError,
        )
    except ImportError as exc:  # pragma: no cover - install-time issue
        raise InvalidToken(
            "PyJWT not installed — pip install pyjwt to enable auth"
        ) from exc

    # Decode kwargs
    options = {"require": ["exp"]}
    decode_kwargs: dict[str, Any] = {
        "algorithms": [config.algorithm],
        "leeway": config.leeway_seconds,
        "options": options,
    }
    if config.issuer:
        decode_kwargs["issuer"] = config.issuer
    if config.audience:
        decode_kwargs["audience"] = config.audience

    # Pick the verification key
    if config.algorithm == "HS256":
        if not config.secret:
            raise InvalidToken("HS256 configured but no secret set")
        key: str = config.secret
    elif config.algorithm == "RS256":
        if not config.public_key:
            raise InvalidToken("RS256 configured but no public_key set")
        key = config.public_key
    else:
        raise InvalidToken(f"unsupported algorithm: {config.algorithm}")

    try:
        payload = pyjwt.decode(token, key, **decode_kwargs)
    except ExpiredSignatureError as exc:
        raise ExpiredToken("token expired") from exc
    except (InvalidSignatureError, InvalidAudienceError, InvalidIssuerError) as exc:
        raise InvalidToken(f"token rejected: {type(exc).__name__}") from exc
    except InvalidTokenError as exc:
        raise InvalidToken(f"token invalid: {exc}") from exc

    tenant = payload.get(config.tenant_claim)
    if not tenant:
        raise InvalidToken(
            f"token missing required '{config.tenant_claim}' claim"
        )

    roles_raw = payload.get(config.roles_claim, [])
    if isinstance(roles_raw, str):
        roles = [roles_raw]
    elif isinstance(roles_raw, list):
        roles = [str(r) for r in roles_raw]
    else:
        roles = []

    return TokenClaims(
        tenant_id=str(tenant),
        user_id=str(payload[config.user_claim]) if config.user_claim in payload else None,
        roles=roles,
        expires_at=payload.get("exp"),
        raw=payload,
    )


# ──────────────────────────────────────────────────────────────
# Env-driven config
# ──────────────────────────────────────────────────────────────
def resolve_jwt_config() -> JWTConfig:
    """Build a ``JWTConfig`` from env vars.

    Env keys (all optional except the algorithm-appropriate one):
      VALONY_JWT_ALGORITHM   default 'HS256'
      VALONY_JWT_SECRET      required when algorithm=HS256
      VALONY_JWT_PUBLIC_KEY  required when algorithm=RS256 (PEM contents)
      VALONY_JWT_ISSUER      validates iss when set
      VALONY_JWT_AUDIENCE    validates aud when set
      VALONY_JWT_LEEWAY      seconds, default 30
      VALONY_JWT_TENANT_CLAIM   default 'tenant_id'
      VALONY_JWT_USER_CLAIM     default 'sub'
      VALONY_JWT_ROLES_CLAIM    default 'roles'
    """
    return JWTConfig(
        algorithm=os.environ.get("VALONY_JWT_ALGORITHM", "HS256"),
        secret=os.environ.get("VALONY_JWT_SECRET") or None,
        public_key=os.environ.get("VALONY_JWT_PUBLIC_KEY") or None,
        issuer=os.environ.get("VALONY_JWT_ISSUER") or None,
        audience=os.environ.get("VALONY_JWT_AUDIENCE") or None,
        leeway_seconds=int(os.environ.get("VALONY_JWT_LEEWAY", "30")),
        tenant_claim=os.environ.get("VALONY_JWT_TENANT_CLAIM", "tenant_id"),
        user_claim=os.environ.get("VALONY_JWT_USER_CLAIM", "sub"),
        roles_claim=os.environ.get("VALONY_JWT_ROLES_CLAIM", "roles"),
    )


# ──────────────────────────────────────────────────────────────
# Synthetic claims for dev / auth-disabled mode
# ──────────────────────────────────────────────────────────────
def public_claims() -> TokenClaims:
    """Default claims used when ``VALONY_AUTH_REQUIRED=0`` (dev mode).

    Endpoints downstream can still read ``request.state.claims.tenant_id``
    and get a stable string — they don't need a separate "auth disabled"
    code path. The tenant string is intentionally distinctive so any
    persisted record clearly came from an unauthenticated dev request.
    """
    return TokenClaims(
        tenant_id="public",
        user_id=None,
        roles=["dev"],
        expires_at=None,
        raw={"_synthetic": True, "reason": "auth_disabled"},
    )
