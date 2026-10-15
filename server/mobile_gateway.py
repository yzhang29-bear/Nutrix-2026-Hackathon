"""Clerk-authenticated mobile facade for the PMRF service."""
import os
from dataclasses import dataclass

import jwt
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class MobileIdentity:
    person_id: str
    role: str


@dataclass(frozen=True)
class ClerkConfig:
    audience: str
    demo_person_ids: frozenset[str]
    issuer: str
    person_id_claim: str
    role_claim: str


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured before enabling mobile routes")
    return value


def clerk_config() -> ClerkConfig:
    demo_person_ids = frozenset(
        person_id.strip() for person_id in _required("MOBILE_DEMO_PERSON_IDS").split(",") if person_id.strip()
    )
    if not demo_person_ids:
        raise RuntimeError("MOBILE_DEMO_PERSON_IDS must contain at least one PMRF person ID")
    return ClerkConfig(
        audience=_required("CLERK_JWT_AUDIENCE"),
        demo_person_ids=demo_person_ids,
        issuer=_required("CLERK_JWT_ISSUER").rstrip("/"),
        person_id_claim=os.environ.get("CLERK_PERSON_ID_CLAIM", "pmrf_person_id"),
        role_claim=os.environ.get("CLERK_ROLE_CLAIM", "role"),
    )


def verify_mobile_request(request: Request) -> MobileIdentity:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "missing bearer token")

    config = clerk_config()
    try:
        signing_key = jwt.PyJWKClient(f"{config.issuer}/.well-known/jwks.json").get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=config.audience,
            issuer=config.issuer,
            options={"require": ["aud", "exp", "iat", "iss", "sub"]},
        )
    except jwt.PyJWTError as error:
        raise HTTPException(401, "invalid or expired session token") from error

    role = claims.get(config.role_claim)
    if not isinstance(role, str) or role.lower() not in {"admin", "clinician", "patient"}:
        raise HTTPException(403, "session token has no authorized Nutrix role")

    person_id = claims.get(config.person_id_claim)
    if not isinstance(person_id, str) or not person_id:
        raise HTTPException(403, "session token has no assigned demo patient")
    if person_id not in config.demo_person_ids:
        raise HTTPException(403, "session token is not assigned to an enabled demo patient")
    return MobileIdentity(person_id=person_id, role=role.lower())
