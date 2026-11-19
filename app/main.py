import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from app.auth import (
    create_access_token,
    create_user_access_token,
    get_current_admin,
    get_current_principal,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database import get_connection, init_db
from app.extractor import extract_text_from_bytes
from app.rag import chunk_text, generate_answer, generate_embedding, search_chunks_by_role

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Modul 2 AI - Custom Chatbot RAG Berbasis Role",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def seed_default_documents() -> None:
    if not DATASET_DIR.exists():
        return

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM documents")
    if cur.fetchone()["cnt"] > 0:
        conn.close()
        return

    cur.execute("SELECT id FROM roles WHERE LOWER(name) = 'mahasiswa'")
    role_row = cur.fetchone()
    if not role_row:
        conn.close()
        return
    role_id = role_row["id"]

    for file_path in sorted(DATASET_DIR.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}:
            continue

        file_bytes = file_path.read_bytes()
        try:
            extracted_text = extract_text_from_bytes(file_bytes, file_path.name)
        except Exception:
            continue
        if not extracted_text:
            continue

        chunks = chunk_text(extracted_text)
        if not chunks:
            continue

        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_path.name)
        stored_filename = f"default_{safe_name}"
        (UPLOAD_DIR / stored_filename).write_bytes(file_bytes)

        doc_title = file_path.stem.replace("_", " ")
        ext = file_path.suffix.lstrip(".").lower() or "txt"
        now = datetime.now(timezone.utc).isoformat()

        cur.execute(
            "INSERT INTO documents (title, filename, file_type, total_chunks, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (doc_title, stored_filename, ext, len(chunks), now),
        )
        doc_id = cur.lastrowid

        cur.execute(
            "INSERT OR IGNORE INTO document_roles (document_id, role_id) VALUES (?, ?)",
            (doc_id, role_id),
        )

        for idx, ch in enumerate(chunks):
            emb = generate_embedding(ch)
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, text, embedding_json) VALUES (?, ?, ?, ?)",
                (doc_id, idx, ch, json.dumps(emb)),
            )

    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_db()
    seed_default_documents()


# ---------- Schemas ----------

class LoginRequest(BaseModel):
    username: str
    password: str


class RoleCreateRequest(BaseModel):
    name: str
    description: str = ""


class RoleUpdateRequest(BaseModel):
    name: str
    description: str = ""


class UserRoleAssignmentRequest(BaseModel):
    role_id: int


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role_id: int


class DocumentRolesRequest(BaseModel):
    role_ids: List[int]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


# ---------- Page Routes ----------

@app.get("/", include_in_schema=False)
def serve_chat_page():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/admin", include_in_schema=False)
def serve_admin_page():
    return FileResponse(str(STATIC_DIR / "admin.html"))


# ---------- Auth Routes ----------

@app.post("/api/auth/login")
def auth_login(payload: LoginRequest):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM admins WHERE username = ?", (payload.username.strip(),))
    admin = cur.fetchone()

    if admin:
        conn.close()
        if not verify_password(payload.password, admin["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Username atau password salah.",
            )
        token = create_access_token(admin["username"], admin["id"])
        return {"success": True, "token": token, "username": admin["username"], "role": "admin", "principal_type": "admin"}

    cur.execute(
        """
        SELECT u.id, u.username, u.password_hash, u.role_id, r.name AS role_name
        FROM users u JOIN roles r ON r.id = u.role_id
        WHERE u.username = ?
        """,
        (payload.username.strip(),),
    )
    user = cur.fetchone()
    conn.close()
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Username atau password salah.")

    token = create_user_access_token(user["username"], user["id"])
    return {
        "success": True,
        "token": token,
        "username": user["username"],
        "role": user["role_name"],
        "principal_type": "user",
    }


@app.get("/api/auth/me")
def get_me(admin: dict = Depends(get_current_admin)):
    return {"success": True, "admin": admin}


@app.get("/api/auth/user-me")
def get_user_me(user: dict = Depends(get_current_user)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.id
        FROM documents d
        JOIN document_roles dr ON dr.document_id = d.id
        WHERE d.is_active = 1 AND dr.role_id = ?
        ORDER BY d.id ASC
        """,
        (user["role_id"],),
    )
    accessible_document_ids = [row["id"] for row in cur.fetchall()]
    conn.close()
    return {
        "success": True,
        "user": {key: user[key] for key in ("user_id", "username", "role_id", "role_name")},
        "accessible_document_ids": accessible_document_ids,
    }


# ---------- Role Routes ----------

@app.get("/api/roles")
def list_roles(_admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.id, r.name, r.description, r.created_at,
               r.is_system,
               COUNT(DISTINCT dr.document_id) AS document_count,
               (SELECT COUNT(*) FROM users u WHERE u.role_id = r.id) AS user_count
        FROM roles r
        LEFT JOIN document_roles dr ON dr.role_id = r.id
        GROUP BY r.id
        ORDER BY r.id ASC
        """
    )
    roles = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {"success": True, "roles": roles}


@app.get("/api/roles/{role_id}/users")
def list_users_for_role(role_id: int, _admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM roles WHERE id = ?", (role_id,))
    role = cur.fetchone()
    if not role:
        conn.close()
        raise HTTPException(status_code=404, detail="Role tidak ditemukan.")
    cur.execute(
        "SELECT id, username, created_at FROM users WHERE role_id = ? ORDER BY id ASC",
        (role_id,),
    )
    users = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {"success": True, "role": dict(role), "users": users, "user_count": len(users)}


@app.post("/api/roles", status_code=status.HTTP_201_CREATED)
def create_role(payload: RoleCreateRequest, _admin: dict = Depends(get_current_admin)):
    role_name = payload.name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,49}", role_name):
        raise HTTPException(
            status_code=400,
            detail="Nama role harus 1-50 karakter dan hanya berisi huruf kecil, angka, '_' atau '-'.",
        )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM roles WHERE LOWER(name) = LOWER(?)", (role_name,))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail=f"Role '{role_name}' sudah ada.")

    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        "INSERT INTO roles (name, description, created_at) VALUES (?, ?, ?)",
        (role_name, payload.description.strip(), now),
    )
    role_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "success": True,
        "role": {"id": role_id, "name": role_name, "description": payload.description.strip()},
    }


@app.put("/api/roles/{role_id}")
def update_role(role_id: int, payload: RoleUpdateRequest, _admin: dict = Depends(get_current_admin)):
    role_name = payload.name.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,49}", role_name):
        raise HTTPException(
            status_code=400,
            detail="Nama role harus 1-50 karakter dan hanya berisi huruf kecil, angka, '_' atau '-'.",
        )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM roles WHERE id = ?", (role_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Role tidak ditemukan.")

    cur.execute("SELECT id FROM roles WHERE LOWER(name) = LOWER(?) AND id != ?", (role_name, role_id))
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail=f"Nama role '{role_name}' sudah digunakan.")

    cur.execute("SELECT is_system FROM roles WHERE id = ?", (role_id,))
    if cur.fetchone()["is_system"]:
        conn.close()
        raise HTTPException(status_code=409, detail="Role sistem tidak dapat diubah.")

    cur.execute(
        "UPDATE roles SET name = ?, description = ? WHERE id = ?",
        (role_name, payload.description.strip(), role_id),
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": "Role berhasil diperbarui."}


@app.delete("/api/roles/{role_id}")
def delete_role(role_id: int, _admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, is_system FROM roles WHERE id = ?", (role_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Role tidak ditemukan.")

    if row["is_system"]:
        conn.close()
        raise HTTPException(status_code=409, detail="Role sistem tidak dapat dihapus.")

    cur.execute("SELECT COUNT(*) FROM users WHERE role_id = ?", (role_id,))
    user_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM document_roles WHERE role_id = ?", (role_id,))
    document_count = cur.fetchone()[0]
    if user_count or document_count:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=(
                f"Role masih digunakan oleh {user_count} pengguna dan {document_count} dokumen. "
                "Pindahkan relasinya terlebih dahulu."
            ),
        )

    try:
        cur.execute("DELETE FROM roles WHERE id = ?", (role_id,))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=409, detail="Role masih digunakan dan tidak dapat dihapus.")
    conn.close()
    return {"success": True, "message": f"Role '{row['name']}' berhasil dihapus."}


@app.get("/api/admin/users")
def list_users(_admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.username, u.created_at, r.id AS role_id, r.name AS role_name
        FROM users u
        JOIN roles r ON r.id = u.role_id
        ORDER BY u.id ASC
        """
    )
    users = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {"success": True, "users": users}


@app.post("/api/admin/users", status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreateRequest, _admin: dict = Depends(get_current_admin)):
    username = payload.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,64}", username):
        raise HTTPException(status_code=400, detail="Username harus 3-64 karakter dan hanya berisi huruf, angka, '.', '_' atau '-'.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password minimal 8 karakter.")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, is_system FROM roles WHERE id = ?", (payload.role_id,))
    role = cur.fetchone()
    if not role:
        conn.close()
        raise HTTPException(status_code=404, detail="Role tidak ditemukan.")
    if role["is_system"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Role sistem tidak dapat diberikan kepada pengguna biasa.")

    cur.execute(
        "SELECT username FROM users WHERE LOWER(username) = LOWER(?) UNION SELECT username FROM admins WHERE LOWER(username) = LOWER(?)",
        (username, username),
    )
    if cur.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Username sudah digunakan.")

    now = datetime.now(timezone.utc).isoformat()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role_id, created_at) VALUES (?, ?, ?, ?)",
            (username, hash_password(payload.password), role["id"], now),
        )
        user_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=409, detail="Username sudah digunakan atau role tidak valid.")
    conn.close()
    return {
        "success": True,
        "user": {"id": user_id, "username": username, "role_id": role["id"], "role_name": role["name"]},
    }


@app.put("/api/admin/users/{user_id}/role")
def assign_user_role(
    user_id: int,
    payload: UserRoleAssignmentRequest,
    _admin: dict = Depends(get_current_admin),
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, role_id FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan.")

    cur.execute("SELECT id, name, is_system FROM roles WHERE id = ?", (payload.role_id,))
    role = cur.fetchone()
    if not role:
        conn.close()
        raise HTTPException(status_code=404, detail="Role tujuan tidak ditemukan.")
    if role["is_system"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Role sistem tidak dapat diberikan kepada pengguna biasa.")

    cur.execute("UPDATE users SET role_id = ? WHERE id = ?", (payload.role_id, user_id))
    conn.commit()
    conn.close()
    return {
        "success": True,
        "user": {"id": user["id"], "username": user["username"]},
        "role": dict(role),
        "message": "Role pengguna berhasil diperbarui oleh admin.",
    }


# ---------- Document Routes (Admin Only) ----------

@app.get("/api/documents")
def list_documents(_admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY id DESC")
    docs = [dict(r) for r in cur.fetchall()]

    for doc in docs:
        cur.execute(
            """
            SELECT r.id, r.name
            FROM roles r
            JOIN document_roles dr ON dr.role_id = r.id
            WHERE dr.document_id = ?
            ORDER BY r.name ASC
            """,
            (doc["id"],),
        )
        doc["roles"] = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {"success": True, "documents": docs}


@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: int, _admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, filename, file_type, total_chunks FROM documents WHERE id = ?", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    cur.execute(
        "SELECT chunk_index, text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
        (doc_id,),
    )
    chunks = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"success": True, "document": dict(doc), "chunks": chunks}


def _parse_role_ids(raw_role_ids: str) -> List[int]:
    try:
        if raw_role_ids.strip().startswith("["):
            values = json.loads(raw_role_ids)
            if not isinstance(values, list):
                raise ValueError("role_ids harus berupa daftar.")
            if any(type(value) is not int for value in values):
                raise ValueError("Setiap role_id harus berupa bilangan bulat.")
            role_ids = values
        else:
            parts = [part.strip() for part in raw_role_ids.split(",") if part.strip()]
            role_ids = [int(part) for part in parts]
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Format role_ids tidak valid.")

    if not role_ids or any(role_id <= 0 for role_id in role_ids):
        raise HTTPException(status_code=400, detail="Pilih minimal satu role yang valid.")
    if len(set(role_ids)) != len(role_ids):
        raise HTTPException(status_code=400, detail="role_ids tidak boleh berisi role duplikat.")
    return role_ids


def _validate_role_ids(cur: sqlite3.Cursor, role_ids: List[int]) -> None:
    placeholders = ",".join("?" for _ in role_ids)
    cur.execute(f"SELECT id FROM roles WHERE id IN ({placeholders})", role_ids)
    found_ids = {row["id"] for row in cur.fetchall()}
    missing_ids = [role_id for role_id in role_ids if role_id not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Role berikut tidak ditemukan: {', '.join(str(role_id) for role_id in missing_ids)}.",
        )


@app.post("/api/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    role_ids: str = Form(...),  # JSON array e.g. "[1, 2]" or comma-separated "1,2"
    _admin: dict = Depends(get_current_admin),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File wajib dipilih.")

    parsed_ids = _parse_role_ids(role_ids)

    file_bytes = await file.read()
    try:
        extracted_text = extract_text_from_bytes(file_bytes, file.filename)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Teks tidak dapat diekstrak dari dokumen ini.")

    chunks = chunk_text(extracted_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Dokumen tidak menghasilkan chunk teks.")

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename)
    stored_filename = f"{int(time.time())}_{safe_name}"

    doc_title = title.strip() or Path(file.filename).stem
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    cur = conn.cursor()

    try:
        _validate_role_ids(cur, parsed_ids)
    except HTTPException:
        conn.close()
        raise

    stored_path = UPLOAD_DIR / stored_filename
    stored_path.write_bytes(file_bytes)

    try:
        cur.execute(
            "INSERT INTO documents (title, filename, file_type, total_chunks, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (doc_title, stored_filename, ext, len(chunks), now),
        )
        doc_id = cur.lastrowid

        for rid in parsed_ids:
            cur.execute(
                "INSERT OR IGNORE INTO document_roles (document_id, role_id) VALUES (?, ?)",
                (doc_id, rid),
            )

        for idx, ch in enumerate(chunks):
            emb = generate_embedding(ch)
            cur.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, text, embedding_json) VALUES (?, ?, ?, ?)",
                (doc_id, idx, ch, json.dumps(emb)),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        stored_path.unlink(missing_ok=True)
        conn.close()
        raise
    conn.close()

    return {
        "success": True,
        "message": f"Dokumen '{doc_title}' berhasil diunggah ({len(chunks)} chunks).",
        "document_id": doc_id,
    }


@app.put("/api/documents/{doc_id}/roles")
def update_document_roles(
    doc_id: int,
    payload: DocumentRolesRequest,
    _admin: dict = Depends(get_current_admin),
):
    role_ids = payload.role_ids
    if not role_ids or any(type(role_id) is not int or role_id <= 0 for role_id in role_ids):
        raise HTTPException(status_code=400, detail="Pilih minimal satu role yang valid.")
    if len(set(role_ids)) != len(role_ids):
        raise HTTPException(status_code=400, detail="role_ids tidak boleh berisi role duplikat.")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM documents WHERE id = ?", (doc_id,))
    document = cur.fetchone()
    if not document:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    try:
        _validate_role_ids(cur, role_ids)
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM document_roles WHERE document_id = ?", (doc_id,))
        cur.executemany(
            "INSERT INTO document_roles (document_id, role_id) VALUES (?, ?)",
            [(doc_id, role_id) for role_id in role_ids],
        )
        conn.commit()
    except HTTPException:
        conn.rollback()
        conn.close()
        raise
    except sqlite3.IntegrityError:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=400, detail="Role tidak valid atau akses dokumen gagal diperbarui.")
    except Exception:
        conn.rollback()
        conn.close()
        raise

    cur.execute(
        "SELECT r.id, r.name FROM roles r JOIN document_roles dr ON dr.role_id = r.id "
        "WHERE dr.document_id = ? ORDER BY r.name ASC",
        (doc_id,),
    )
    roles = [dict(row) for row in cur.fetchall()]
    conn.close()
    return {
        "success": True,
        "document": {"id": document["id"], "title": document["title"]},
        "roles": roles,
        "message": "Hak akses dokumen berhasil diperbarui.",
    }


@app.patch("/api/documents/{doc_id}/toggle")
def toggle_document_status(doc_id: int, _admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, is_active FROM documents WHERE id = ?", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    new_status = 0 if doc["is_active"] else 1
    cur.execute("UPDATE documents SET is_active = ? WHERE id = ?", (new_status, doc_id))
    conn.commit()
    conn.close()

    state_label = "diaktifkan" if new_status == 1 else "dinonaktifkan"
    return {
        "success": True,
        "is_active": bool(new_status),
        "message": f"Dokumen '{doc['title']}' berhasil {state_label}.",
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, _admin: dict = Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, title FROM documents WHERE id = ?", (doc_id,))
    doc = cur.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

    file_path = UPLOAD_DIR / doc["filename"]
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    return {"success": True, "message": f"Dokumen '{doc['title']}' berhasil dihapus."}


# ---------- Authenticated Chat Route (Role-Filtered) ----------

@app.post("/api/chat")
def chat_with_rag(payload: ChatRequest, principal: dict = Depends(get_current_principal)):
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    role_id = principal["role_id"] if principal["principal_type"] == "user" else None
    role_name = principal["role_name"]
    relevant_chunks = search_chunks_by_role(question, role_id, top_k=7, min_score=0.16)
    result = generate_answer(question, role_name, relevant_chunks)

    return {
        "success": True,
        "role": role_name,
        "answer": result["answer"],
        "sources": result["sources"],
    }


def _parse_duration_seconds(dur: str) -> float:
    if not dur:
        return 0.0
    total = 0.0
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", dur.lower()):
        num = float(val)
        if unit == "h":
            total += num * 3600
        elif unit == "m":
            total += num * 60
        elif unit == "s":
            total += num
        elif unit == "ms":
            total += num / 1000.0
    return total


def _format_remaining_duration(seconds: float) -> str:
    if seconds <= 0:
        return "Sudah reset"
    secs = int(round(seconds))
    if secs < 60:
        return f"{secs}d"
    mins, rem_s = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {rem_s}d" if rem_s else f"{mins}m"
    hours, rem_m = divmod(mins, 60)
    return f"{hours}j {rem_m}m"


@app.get("/api/admin/groq-quota")
def get_groq_quota(_admin: dict = Depends(get_current_admin)):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    configured_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    has_api_key = bool(api_key and api_key != "your_groq_api_key_here")

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT model, limit_requests, remaining_requests, reset_requests,
               limit_tokens, remaining_tokens, reset_tokens, created_at
        FROM groq_usage_logs
        WHERE limit_requests IS NOT NULL OR limit_tokens IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
        """
    )
    latest_row = cur.fetchone()
    latest = dict(latest_row) if latest_row else None

    if latest:
        try:
            recorded_at = datetime.fromisoformat(latest["created_at"])
            elapsed = max(0.0, (datetime.now(timezone.utc) - recorded_at).total_seconds())
        except Exception:
            elapsed = 0.0

        req_reset_sec = _parse_duration_seconds(latest.get("reset_requests", ""))
        req_rem_sec = max(0.0, req_reset_sec - elapsed)
        if latest.get("limit_requests") and req_reset_sec > 0 and req_rem_sec <= 0:
            latest["remaining_requests"] = latest["limit_requests"]
            latest["reset_requests_label"] = "Sudah reset"
        else:
            latest["reset_requests_label"] = _format_remaining_duration(req_rem_sec) if req_reset_sec > 0 else "-"

        tok_reset_sec = _parse_duration_seconds(latest.get("reset_tokens", ""))
        tok_rem_sec = max(0.0, tok_reset_sec - elapsed)
        if latest.get("limit_tokens") and tok_reset_sec > 0 and tok_rem_sec <= 0:
            latest["remaining_tokens"] = latest["limit_tokens"]
            latest["reset_tokens_label"] = "Sudah reset"
        else:
            latest["reset_tokens_label"] = _format_remaining_duration(tok_rem_sec) if tok_reset_sec > 0 else "-"

    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur.execute(
        """
        SELECT COUNT(*) AS requests_today,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens_today,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens_today,
               COALESCE(SUM(total_tokens), 0) AS total_tokens_today
        FROM groq_usage_logs
        WHERE created_at LIKE ?
        """,
        (f"{today_prefix}%",),
    )
    today_row = dict(cur.fetchone())

    cur.execute(
        """
        SELECT COUNT(*) AS total_requests,
               COALESCE(SUM(total_tokens), 0) AS total_tokens
        FROM groq_usage_logs
        """
    )
    total_row = dict(cur.fetchone())

    cur.execute(
        """
        SELECT id, model, prompt_tokens, completion_tokens, total_tokens, status, created_at
        FROM groq_usage_logs
        ORDER BY id DESC
        LIMIT 10
        """
    )
    recent_logs = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {
        "success": True,
        "has_api_key": has_api_key,
        "configured_model": configured_model,
        "latest": latest,
        "today": today_row,
        "total": total_row,
        "recent_logs": recent_logs,
    }


def _get_accessible_document(doc_id: int, principal: dict, cur: sqlite3.Cursor):
    if principal["principal_type"] == "admin":
        cur.execute(
            "SELECT id, title, filename, file_type, total_chunks, is_active FROM documents WHERE id = ?",
            (doc_id,),
        )
    else:
        cur.execute(
            """
            SELECT d.id, d.title, d.filename, d.file_type, d.total_chunks, d.is_active
            FROM documents d
            JOIN document_roles dr ON dr.document_id = d.id
            WHERE d.id = ? AND d.is_active = 1 AND dr.role_id = ?
            """,
            (doc_id, principal["role_id"]),
        )
    return cur.fetchone()


@app.get("/api/documents/{doc_id}/source")
@app.get("/api/public/documents/{doc_id}")
def get_document_source(doc_id: int, principal: dict = Depends(get_current_principal)):
    conn = get_connection()
    cur = conn.cursor()
    doc = _get_accessible_document(doc_id, principal, cur)
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    file_path = UPLOAD_DIR / doc["filename"]
    full_text = ""
    if file_path.exists():
        try:
            full_text = extract_text_from_bytes(file_path.read_bytes(), doc["filename"])
        except Exception:
            full_text = ""

    if not full_text:
        cur.execute(
            "SELECT text FROM document_chunks WHERE document_id = ? ORDER BY chunk_index ASC",
            (doc_id,),
        )
        full_text = "\n\n".join(r["text"] for r in cur.fetchall())

    conn.close()

    return {
        "success": True,
        "document": {key: doc[key] for key in ("id", "title", "filename", "file_type", "total_chunks")},
        "url": f"/api/documents/{doc_id}/file",
        "file_url": f"/api/documents/{doc_id}/file",
        "content": full_text,
    }


@app.get("/api/documents/{doc_id}/file")
def download_document_file(doc_id: int, principal: dict = Depends(get_current_principal)):
    conn = get_connection()
    cur = conn.cursor()
    doc = _get_accessible_document(doc_id, principal, cur)
    conn.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Dokumen tidak ditemukan.")

    file_path = UPLOAD_DIR / doc["filename"]
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File dokumen tidak ditemukan.")
    return FileResponse(file_path, filename=doc["filename"])
