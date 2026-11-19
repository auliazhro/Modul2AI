import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "rahasia_modul2_jwt_2026")
JWT_ALGORITHM = "HS256"
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return hmac.compare_digest(dk.hex(), expected_hex)


def create_access_token(username: str, admin_id: int) -> str:
    payload = {
        "sub": username,
        "admin_id": admin_id,
        "role": "admin",
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_user_access_token(username: str, user_id: int) -> str:
    payload = {
        "sub": username,
        "user_id": user_id,
        "token_type": "user",
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token autentikasi tidak ditemukan.")

    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesi login telah berakhir.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token tidak valid.")

    from app.database import get_connection

    conn = get_connection()
    cur = conn.cursor()

    # Admin memiliki akses administratif penuh; identitasnya tetap dicocokkan
    # ke SQLite agar token akun yang sudah dihapus tidak tetap berlaku.
    if payload.get("role") == "admin" and payload.get("admin_id"):
        cur.execute("SELECT id, username FROM admins WHERE id = ?", (payload["admin_id"],))
        admin = cur.fetchone()
        conn.close()
        if not admin or admin["username"] != payload.get("sub"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Akun admin tidak lagi tersedia.")
        return {
            "principal_type": "admin",
            "admin_id": admin["id"],
            "username": admin["username"],
            "role_id": None,
            "role_name": "admin",
        }

    if payload.get("token_type") != "user" or not isinstance(payload.get("user_id"), int):
        conn.close()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token pengguna tidak valid.")

    cur.execute(
        """
        SELECT u.id, u.username, u.role_id, r.name AS role_name
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE u.id = ?
        """,
        (payload["user_id"],),
    )
    user = cur.fetchone()
    conn.close()
    if not user or user["username"] != payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Akun pengguna tidak memiliki role aktif.")

    return {
        "principal_type": "user",
        "user_id": user["id"],
        "username": user["username"],
        "role_id": user["role_id"],
        "role_name": user["role_name"],
    }


def get_current_user(principal: dict = Depends(get_current_principal)) -> dict:
    if principal["principal_type"] != "user":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Endpoint ini memerlukan akun pengguna.")
    return principal


def get_current_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token admin tidak ditemukan. Silakan login terlebih dahulu.",
        )
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Akses khusus admin.")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesi login telah berakhir.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token tidak valid.")
