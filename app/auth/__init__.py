"""
app/auth — A6a JWT authentication + tenant context.

Public surface:
    JWTConfig          Algorithm + secret/key + issuer/audience + leeway
    TokenClaims        Validated claims with first-class tenant_id / user_id / roles
    decode_token       Pure validator (raises on bad/expired/missing)
    resolve_jwt_config Build config from env vars
    AuthError          Base for InvalidToken / ExpiredToken / MissingToken
    auth_middleware    ASGI middleware: validate, attach claims, reject 401
    public_claims      Default synthetic claims used when auth is disabled
    is_auth_required   Resolves VALONY_AUTH_REQUIRED env var
"""
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
from .middleware import (
    auth_middleware,
    is_auth_required,
)

__all__ = [
    "AuthError",
    "ExpiredToken",
    "InvalidToken",
    "JWTConfig",
    "MissingToken",
    "TokenClaims",
    "decode_token",
    "public_claims",
    "resolve_jwt_config",
    "auth_middleware",
    "is_auth_required",
]
