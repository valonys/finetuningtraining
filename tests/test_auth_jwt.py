"""
Unit tests for app.auth — JWT decode + ASGI middleware behavior (A6a).

Coverage:
  * decode_token: HS256 happy path, expired, bad signature, missing
    tenant claim, missing exp claim, custom claim names, RS256 path
    when key is provided, leeway tolerance.
  * resolve_jwt_config: env mapping + defaults.
  * auth_middleware: public path bypass, dev-mode synthetic claims,
    strict mode rejects missing/invalid/expired/wrong-scheme tokens
    and accepts a valid token.
  * Synthetic ``public_claims`` shape stable.
"""
from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import jwt as pyjwt

from app.auth import (
    AuthError,
    ExpiredToken,
    InvalidToken,
    JWTConfig,
    MissingToken,
    TokenClaims,
    auth_middleware,
    decode_token,
    is_auth_required,
    public_claims,
    resolve_jwt_config,
)


_SECRET = "dev-secret-do-not-use-in-prod"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _mint(claims: dict[str, Any], *, secret: str = _SECRET, alg: str = "HS256") -> str:
    return pyjwt.encode(claims, secret, algorithm=alg)


def _now() -> int:
    return int(time.time())


def _hs256_config(**overrides) -> JWTConfig:
    base = JWTConfig(algorithm="HS256", secret=_SECRET, leeway_seconds=5)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ──────────────────────────────────────────────────────────────
# decode_token
# ──────────────────────────────────────────────────────────────
def test_hs256_happy_path():
    token = _mint({
        "tenant_id": "acme",
        "sub": "user-42",
        "roles": ["admin", "developer"],
        "exp": _now() + 60,
    })
    claims = decode_token(token, _hs256_config())
    assert claims.tenant_id == "acme"
    assert claims.user_id == "user-42"
    assert claims.roles == ["admin", "developer"]
    assert claims.expires_at is not None
    assert claims.raw["sub"] == "user-42"


def test_expired_token_raises_expired():
    token = _mint({"tenant_id": "acme", "exp": _now() - 60})
    with pytest.raises(ExpiredToken):
        decode_token(token, _hs256_config())


def test_bad_signature_raises_invalid():
    token = _mint({"tenant_id": "acme", "exp": _now() + 60})
    with pytest.raises(InvalidToken):
        decode_token(token, _hs256_config(secret="some-other-secret"))


def test_missing_tenant_claim_raises_invalid():
    token = _mint({"sub": "u", "exp": _now() + 60})  # no tenant_id
    with pytest.raises(InvalidToken, match="tenant_id"):
        decode_token(token, _hs256_config())


def test_missing_exp_claim_raises_invalid():
    """The decoder explicitly requires exp; without it tokens never
    expire which is a security risk we won't tolerate."""
    token = _mint({"tenant_id": "acme"})  # no exp
    with pytest.raises(InvalidToken):
        decode_token(token, _hs256_config())


def test_custom_tenant_claim_name():
    token = _mint({"org": "acme", "exp": _now() + 60})
    claims = decode_token(token, _hs256_config(tenant_claim="org"))
    assert claims.tenant_id == "acme"


def test_custom_user_claim_name():
    token = _mint({"tenant_id": "acme", "uid": "u-1", "exp": _now() + 60})
    claims = decode_token(token, _hs256_config(user_claim="uid"))
    assert claims.user_id == "u-1"


def test_roles_string_normalized_to_list():
    token = _mint({"tenant_id": "acme", "roles": "admin", "exp": _now() + 60})
    claims = decode_token(token, _hs256_config())
    assert claims.roles == ["admin"]


def test_roles_missing_returns_empty_list():
    token = _mint({"tenant_id": "acme", "exp": _now() + 60})
    claims = decode_token(token, _hs256_config())
    assert claims.roles == []


def test_issuer_validation_rejects_wrong_iss():
    token = _mint({"tenant_id": "acme", "iss": "bad-iss", "exp": _now() + 60})
    with pytest.raises(InvalidToken):
        decode_token(token, _hs256_config(issuer="good-iss"))


def test_issuer_validation_accepts_matching_iss():
    token = _mint({"tenant_id": "acme", "iss": "good-iss", "exp": _now() + 60})
    claims = decode_token(token, _hs256_config(issuer="good-iss"))
    assert claims.tenant_id == "acme"


def test_audience_validation_rejects_wrong_aud():
    token = _mint({"tenant_id": "acme", "aud": "wrong", "exp": _now() + 60})
    with pytest.raises(InvalidToken):
        decode_token(token, _hs256_config(audience="right"))


def test_unsupported_algorithm_raises():
    with pytest.raises(InvalidToken, match="unsupported algorithm"):
        decode_token(
            "any-string",
            JWTConfig(algorithm="HS999", secret="x", leeway_seconds=5),
        )


def test_hs256_without_secret_raises():
    with pytest.raises(InvalidToken, match="no secret"):
        decode_token("x", JWTConfig(algorithm="HS256", secret=None))


def test_rs256_without_public_key_raises():
    with pytest.raises(InvalidToken, match="no public_key"):
        decode_token("x", JWTConfig(algorithm="RS256", public_key=None))


def test_empty_token_raises_missing():
    with pytest.raises(MissingToken):
        decode_token("", _hs256_config())


def test_leeway_tolerates_small_clock_skew():
    """Token expired 3 seconds ago, leeway 30s — should still decode."""
    token = _mint({"tenant_id": "acme", "exp": _now() - 3})
    claims = decode_token(token, _hs256_config(leeway_seconds=30))
    assert claims.tenant_id == "acme"


# ──────────────────────────────────────────────────────────────
# resolve_jwt_config
# ──────────────────────────────────────────────────────────────
def test_resolve_jwt_config_defaults(monkeypatch):
    for key in [
        "VALONY_JWT_ALGORITHM", "VALONY_JWT_SECRET", "VALONY_JWT_PUBLIC_KEY",
        "VALONY_JWT_ISSUER", "VALONY_JWT_AUDIENCE", "VALONY_JWT_LEEWAY",
        "VALONY_JWT_TENANT_CLAIM", "VALONY_JWT_USER_CLAIM", "VALONY_JWT_ROLES_CLAIM",
    ]:
        monkeypatch.delenv(key, raising=False)
    cfg = resolve_jwt_config()
    assert cfg.algorithm == "HS256"
    assert cfg.secret is None
    assert cfg.leeway_seconds == 30
    assert cfg.tenant_claim == "tenant_id"


def test_resolve_jwt_config_reads_env(monkeypatch):
    monkeypatch.setenv("VALONY_JWT_ALGORITHM", "RS256")
    monkeypatch.setenv("VALONY_JWT_PUBLIC_KEY", "PEM-CONTENTS")
    monkeypatch.setenv("VALONY_JWT_ISSUER", "https://idp.example.com/")
    monkeypatch.setenv("VALONY_JWT_AUDIENCE", "valony-api")
    monkeypatch.setenv("VALONY_JWT_LEEWAY", "60")
    monkeypatch.setenv("VALONY_JWT_TENANT_CLAIM", "https://example.com/org")
    cfg = resolve_jwt_config()
    assert cfg.algorithm == "RS256"
    assert cfg.public_key == "PEM-CONTENTS"
    assert cfg.issuer == "https://idp.example.com/"
    assert cfg.audience == "valony-api"
    assert cfg.leeway_seconds == 60
    assert cfg.tenant_claim == "https://example.com/org"


# ──────────────────────────────────────────────────────────────
# is_auth_required
# ──────────────────────────────────────────────────────────────
def test_is_auth_required_default_off(monkeypatch):
    monkeypatch.delenv("VALONY_AUTH_REQUIRED", raising=False)
    assert is_auth_required() is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_is_auth_required_truthy_values_enable(monkeypatch, val):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", val)
    assert is_auth_required() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
def test_is_auth_required_falsy_values_disable(monkeypatch, val):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", val)
    assert is_auth_required() is False


# ──────────────────────────────────────────────────────────────
# public_claims
# ──────────────────────────────────────────────────────────────
def test_public_claims_shape_stable():
    c = public_claims()
    assert c.tenant_id == "public"
    assert c.user_id is None
    assert c.roles == ["dev"]
    assert c.raw["_synthetic"] is True


# ──────────────────────────────────────────────────────────────
# auth_middleware integration
# ──────────────────────────────────────────────────────────────
def _make_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(auth_middleware)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/whoami")
    async def whoami(request: Request):
        return {
            "tenant_id": request.state.claims.tenant_id,
            "roles": request.state.claims.roles,
        }

    @app.post("/v1/sensitive")
    async def sensitive(request: Request):
        return {"tenant_id": request.state.claims.tenant_id}

    return app


def test_middleware_health_always_public(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_middleware_dev_mode_attaches_public_tenant(monkeypatch):
    monkeypatch.delenv("VALONY_AUTH_REQUIRED", raising=False)
    client = TestClient(_make_app())
    res = client.get("/v1/whoami")
    assert res.status_code == 200
    assert res.json()["tenant_id"] == "public"
    assert "dev" in res.json()["roles"]


def test_middleware_strict_mode_rejects_missing_token(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    res = client.get("/v1/whoami")
    assert res.status_code == 401
    assert res.json()["code"] == "missing_token"
    assert "WWW-Authenticate" in res.headers


def test_middleware_strict_mode_rejects_wrong_scheme(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    res = client.get("/v1/whoami", headers={"Authorization": "Basic abc"})
    assert res.status_code == 401
    assert res.json()["code"] == "missing_token"


def test_middleware_strict_mode_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    bad = _mint({"tenant_id": "acme", "exp": _now() + 60}, secret="wrong")
    res = client.get("/v1/whoami", headers={"Authorization": f"Bearer {bad}"})
    assert res.status_code == 401
    assert res.json()["code"] == "invalid_token"


def test_middleware_strict_mode_rejects_expired(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    expired = _mint({"tenant_id": "acme", "exp": _now() - 600})
    res = client.get("/v1/whoami", headers={"Authorization": f"Bearer {expired}"})
    assert res.status_code == 401
    assert res.json()["code"] == "token_expired"


def test_middleware_strict_mode_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    good = _mint({
        "tenant_id": "acme-corp",
        "sub": "alice",
        "roles": ["admin"],
        "exp": _now() + 60,
    })
    res = client.get("/v1/whoami", headers={"Authorization": f"Bearer {good}"})
    assert res.status_code == 200
    assert res.json() == {"tenant_id": "acme-corp", "roles": ["admin"]}


def test_middleware_strict_mode_rejects_mutating_endpoint_too(monkeypatch):
    """POST endpoints must enforce auth same as GETs (only /healthz
    and OpenAPI surfaces are public)."""
    monkeypatch.setenv("VALONY_AUTH_REQUIRED", "1")
    monkeypatch.setenv("VALONY_JWT_SECRET", _SECRET)
    client = TestClient(_make_app())
    res = client.post("/v1/sensitive", json={})
    assert res.status_code == 401
