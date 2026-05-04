"""
app/auth/middleware.py
──────────────────────
ASGI middleware that validates Bearer tokens and stashes claims on
``request.state.claims`` (A6a).

Why a middleware vs per-route ``Depends``: applying auth uniformly
across 30+ endpoints via dependencies means touching every signature.
A middleware applies once, cannot be forgotten when adding a new
route, and keeps each endpoint's signature focused on its actual
business inputs.

Activation:
  * ``VALONY_AUTH_REQUIRED=1`` (or true / yes / on) — strict mode:
    every request except a small allowlist (``/healthz``, OpenAPI
    surfaces) must carry a valid Bearer JWT. Missing or invalid
    tokens get 401 with a JSON detail.
  * Anything else (default) — auth disabled. The middleware still
    runs but attaches synthetic ``public_claims()`` so downstream
    handlers can rely on ``request.state.claims`` always being set.

This dual mode keeps the existing dev workflow working untouched
while making prod opt in to enforcement via env. The deploy guide
(``docs/DEPLOY_APPRUNNER.md``) instructs the operator to set
``VALONY_AUTH_REQUIRED=1`` along with the JWT secret.
"""
from __future__ import annotations

import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from .jwt import (
    AuthError,
    ExpiredToken,
    InvalidToken,
    JWTConfig,
    MissingToken,
    TokenClaims,
    decode_token,
    public_claims,
    resolve_jwt_config,
)

logger = logging.getLogger(__name__)


# Paths that bypass auth entirely. Health checks must remain
# unauthenticated (otherwise infrastructure can't probe liveness),
# and the OpenAPI / docs surfaces are public schema for tooling.
_PUBLIC_PATHS = frozenset({
    "/healthz",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
})


def is_auth_required() -> bool:
    """Resolve the env flag once per call. ``"1" / "true" / "yes" / "on"``
    enable; anything else (including unset) keeps auth disabled."""
    raw = os.environ.get("VALONY_AUTH_REQUIRED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _extract_bearer(request: Request) -> str:
    """Pull the token out of the ``Authorization: Bearer ...`` header.
    Raises ``MissingToken`` if absent or wrong scheme."""
    header = request.headers.get("authorization")
    if not header:
        raise MissingToken("missing Authorization header")
    parts = header.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise MissingToken("Authorization header must be 'Bearer <token>'")
    token = parts[1].strip()
    if not token:
        raise MissingToken("empty bearer token")
    return token


async def auth_middleware(request: Request, call_next):
    """ASGI middleware entrypoint.

    The config is resolved per request so test monkeypatches of env
    vars take effect without process restart. In production (where
    env is stable) this is a couple of dict lookups — negligible cost.
    """
    # 1. Public bypass for health + OpenAPI surfaces.
    if request.url.path in _PUBLIC_PATHS:
        request.state.claims = public_claims()
        return await call_next(request)

    # 2. Auth disabled — attach synthetic claims and continue.
    if not is_auth_required():
        request.state.claims = public_claims()
        return await call_next(request)

    # 3. Strict mode — extract + validate.
    config = resolve_jwt_config()
    try:
        token = _extract_bearer(request)
        claims = decode_token(token, config)
    except MissingToken as exc:
        return _unauthorized(str(exc), code="missing_token")
    except ExpiredToken as exc:
        return _unauthorized(str(exc), code="token_expired")
    except InvalidToken as exc:
        return _unauthorized(str(exc), code="invalid_token")
    except AuthError as exc:                    # safety net
        return _unauthorized(str(exc), code="auth_error")

    request.state.claims = claims
    return await call_next(request)


def _unauthorized(detail: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": detail, "code": code},
        headers={"WWW-Authenticate": 'Bearer realm="valonylabs"'},
    )


def get_claims(request: Request) -> TokenClaims:
    """Convenience accessor for endpoints that want the typed object.

    Returns the synthetic public claims if the middleware hasn't run
    (shouldn't happen in production but keeps unit tests of single
    endpoints from crashing on missing state)."""
    return getattr(request.state, "claims", public_claims())
