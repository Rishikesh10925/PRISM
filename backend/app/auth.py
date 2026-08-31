"""Admin authentication (Phase 8 hardening): the worklist-management endpoints require a
valid bearer token, obtained via POST /api/auth/login with the admin username/password
configured through ADMIN_USERNAME / ADMIN_PASSWORD_HASH (see .env.example). Citizen-facing
reporting (POST /api/report) stays open -- anyone can report a pothole, only the
authenticated municipal admin can see or manage the worklist.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from app.config import ADMIN_PASSWORD_HASH, ADMIN_USERNAME, JWT_ALGORITHM, JWT_EXPIRE_MINUTES, JWT_SECRET

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def verify_admin_credentials(username: str, password: str) -> bool:
    if username != ADMIN_USERNAME:
        # Still run a hash verify on a bogus value so a wrong-username request takes
        # roughly the same time as a wrong-password one (avoid trivial user enumeration
        # via response timing).
        pwd_context.verify(password, ADMIN_PASSWORD_HASH)
        return False
    return pwd_context.verify(password, ADMIN_PASSWORD_HASH)


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_admin(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token")
    return payload["sub"]
