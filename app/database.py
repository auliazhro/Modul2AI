import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "modul2ai.db")


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    from app.auth import hash_password

    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            is_system INTEGER NOT NULL DEFAULT 0 CHECK (is_system IN (0, 1))
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            total_chunks INTEGER DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            uploaded_at TEXT NOT NULL
        );
    """)

    cur.execute("PRAGMA table_info(documents)")
    doc_cols = {row["name"] for row in cur.fetchall()}
    if "is_active" not in doc_cols:
        cur.execute("ALTER TABLE documents ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_roles (
            document_id INTEGER NOT NULL,
            role_id INTEGER NOT NULL,
            PRIMARY KEY (document_id, role_id),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE RESTRICT
        );
    """)

    # Akun non-admin terikat tepat pada satu role. Admin yang sudah ada tetap
    # disimpan di tabel admins dan tetap memakai mekanisme JWT yang sekarang.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE RESTRICT
        );
    """)

    cur.execute("PRAGMA table_info(roles)")
    role_cols = {row["name"] for row in cur.fetchall()}
    if "is_system" not in role_cols:
        cur.execute("ALTER TABLE roles ADD COLUMN is_system INTEGER NOT NULL DEFAULT 0 CHECK (is_system IN (0, 1))")

    cur.execute("PRAGMA foreign_keys = OFF")
    cur.execute("PRAGMA foreign_key_list(document_roles)")
    role_fks = cur.fetchall()
    fk_actions = {row["table"]: row["on_delete"].upper() for row in role_fks}
    if fk_actions != {"documents": "CASCADE", "roles": "RESTRICT"}:
        cur.execute("BEGIN")
        cur.execute("""
            CREATE TABLE document_roles_migrating (
                document_id INTEGER NOT NULL,
                role_id INTEGER NOT NULL,
                PRIMARY KEY (document_id, role_id),
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE RESTRICT
            )
        """)
        cur.execute("INSERT INTO document_roles_migrating (document_id, role_id) SELECT document_id, role_id FROM document_roles")
        cur.execute("DROP TABLE document_roles")
        cur.execute("ALTER TABLE document_roles_migrating RENAME TO document_roles")
        cur.execute("COMMIT")
    cur.execute("PRAGMA foreign_keys = ON")

    cur.execute(
        "INSERT OR IGNORE INTO roles (name, description, created_at, is_system) VALUES (?, ?, ?, 1)",
        ("admin", "Role administratif sistem", now),
    )
    cur.execute("UPDATE roles SET is_system = 1 WHERE name = 'admin'")
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS prevent_system_role_delete
        BEFORE DELETE ON roles
        WHEN OLD.is_system = 1
        BEGIN
            SELECT RAISE(ABORT, 'Role sistem tidak dapat dihapus');
        END
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS groq_usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            limit_requests INTEGER,
            remaining_requests INTEGER,
            reset_requests TEXT DEFAULT '',
            limit_tokens INTEGER,
            remaining_tokens INTEGER,
            reset_tokens TEXT DEFAULT '',
            status TEXT DEFAULT 'success',
            created_at TEXT NOT NULL
        );
    """)

    default_roles = [
        ("mahasiswa", "Akses informasi akademik & layanan SSC untuk mahasiswa"),
    ]
    for role_name, desc in default_roles:
        cur.execute(
            "INSERT OR IGNORE INTO roles (name, description, created_at) VALUES (?, ?, ?)",
            (role_name, desc, now),
        )

    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    cur.execute("SELECT id FROM admins WHERE username = ?", (admin_username,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
            (admin_username, hash_password(admin_password), now),
        )

    conn.commit()
    conn.close()
