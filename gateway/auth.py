"""Authentication dependency for Cinch FastAPI Gateway."""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from gateway.config import GatewaySettings, get_settings


def get_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    current_settings: GatewaySettings = Depends(get_settings),
) -> Optional[str]:
    """Validate request API key against configured secret.

    Supports both 'Authorization: Bearer <token>' and 'X-API-Key: <token>' headers.
    If no API key is configured on the gateway, authentication passes transparently.
    """
    configured_key = current_settings.gateway_api_key
    if not configured_key:
        # Auth disabled / not configured
        return None

    extracted_key: Optional[str] = None
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            extracted_key = parts[1]
        elif len(parts) == 1:
            extracted_key = parts[0]

    if not extracted_key and x_api_key:
        extracted_key = x_api_key.strip()

    if not extracted_key or not secrets.compare_digest(extracted_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return extracted_key
