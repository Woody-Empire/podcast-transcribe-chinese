import hashlib
import secrets
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "podcast.db"
PODCASTS_DIR = DATA_DIR / "podcasts"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    PODCASTS_DIR.mkdir(exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS podcasts (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS playback_progress (
            user_id INTEGER NOT NULL,
            podcast_id TEXT NOT NULL,
            position REAL NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, podcast_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (podcast_id) REFERENCES podcasts(id)
        );
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()


def create_user(username: str, password: str) -> int | None:
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, pw_hash, salt),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def verify_user(username: str, password: str) -> int | None:
    conn = get_db()
    row = conn.execute(
        "SELECT id, password_hash, salt FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    if _hash_password(password, row["salt"]) == row["password_hash"]:
        return row["id"]
    return None


def create_session(user_id: int) -> str:
    token = secrets.token_hex(32)
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (token, user_id) VALUES (?, ?)", (token, user_id)
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_token(token: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT u.id, u.username FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (token,),
    ).fetchone()
    conn.close()
    if row:
        return {"id": row["id"], "username": row["username"]}
    return None


def delete_session(token: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def save_podcast(podcast_id: str, user_id: int, title: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO podcasts (id, user_id, title) VALUES (?, ?, ?)",
        (podcast_id, user_id, title),
    )
    conn.commit()
    conn.close()


def get_user_podcasts(user_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        """SELECT p.id, p.title, p.created_at,
                  pp.position, pp.updated_at as last_played_at
           FROM podcasts p
           LEFT JOIN playback_progress pp ON p.id = pp.podcast_id AND pp.user_id = ?
           WHERE p.user_id = ?
           ORDER BY COALESCE(pp.updated_at, p.created_at) DESC""",
        (user_id, user_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_progress(user_id: int, podcast_id: str, position: float):
    conn = get_db()
    conn.execute(
        """INSERT INTO playback_progress (user_id, podcast_id, position, updated_at)
           VALUES (?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT (user_id, podcast_id)
           DO UPDATE SET position = excluded.position, updated_at = CURRENT_TIMESTAMP""",
        (user_id, podcast_id, position),
    )
    conn.commit()
    conn.close()


def get_last_played(user_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute(
        """SELECT p.id as podcast_id, p.title, pp.position
           FROM playback_progress pp
           JOIN podcasts p ON pp.podcast_id = p.id
           WHERE pp.user_id = ?
           ORDER BY pp.updated_at DESC
           LIMIT 1""",
        (user_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_progress(user_id: int, podcast_id: str) -> float:
    conn = get_db()
    row = conn.execute(
        "SELECT position FROM playback_progress WHERE user_id = ? AND podcast_id = ?",
        (user_id, podcast_id),
    ).fetchone()
    conn.close()
    return row["position"] if row else 0.0
