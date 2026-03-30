"""Simple API key auth — issues JWT-like tokens for dashboard access."""
import os
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/auth", tags=["auth"])

ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "watchlog-demo")


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def get_token(body: TokenRequest):
    if body.api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Return the key itself as the token (stateless, simple MVP auth)
    return TokenResponse(access_token=body.api_key)
