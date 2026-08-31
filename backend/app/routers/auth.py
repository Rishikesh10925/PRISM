"""POST /api/auth/login: exchange the admin username/password for a JWT bearer token."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.auth import create_access_token, verify_admin_credentials

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    if not verify_admin_credentials(body.username, body.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="incorrect username or password")
    return TokenResponse(access_token=create_access_token(body.username))
