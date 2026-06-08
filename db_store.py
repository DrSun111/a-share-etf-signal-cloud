from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
LOCAL_DB_PATH = APP_DIR / "data" / "etf_signal.db"


def _secret_database_url() -> str | None:
    try:
        import streamlit as st

        if "DATABASE_URL" in st.secrets:
            return str(st.secrets["DATABASE_URL"])
        if "database" in st.secrets and "url" in st.secrets["database"]:
            return str(st.secrets["database"]["url"])
    except Exception:
        return None
    return None


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or _secret_database_url() or f"sqlite:///{LOCAL_DB_PATH.as_posix()}"


def is_sqlite_url(url: str | None = None) -> bool:
    url = url or database_url()
    return url.startswith("sqlite:///")


def masked_database_url(url: str | None = None) -> str:
    url = url or database_url()
    if is_sqlite_url(url):
        return url
    if "@" not in url or "://" not in url:
        return "***"
    scheme, rest = url.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


def _sqlite_path(url: str) -> Path:
    return Path(url.replace("sqlite:///", "", 1))


def _sqlite_conn(url: str | None = None) -> sqlite3.Connection:
    url = url or database_url()
    path = _sqlite_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def _pg_engine(url: str | None = None):
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError("PostgreSQL 模式需要安装 sqlalchemy 和 psycopg[binary]。") from exc

    url = url or database_url()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, pool_pre_ping=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _execute(sql: str, params: tuple[Any, ...] | dict[str, Any] | None = None) -> None:
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            conn.execute(sql, params or ())
            conn.commit()
        return

    from sqlalchemy import text

    engine = _pg_engine(url)
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def init_db() -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS etf_spot_snapshot (
            snapshot_ts TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            payload_json TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sector_heat_snapshot (
            snapshot_ts TEXT NOT NULL,
            sector TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS score_snapshot (
            snapshot_ts TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            total_score REAL,
            action TEXT,
            payload_json TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            owner TEXT NOT NULL,
            code TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
    ]
    for statement in statements:
        _execute(statement)


def _json_ready_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _insert_rows(table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    url = database_url()
    if is_sqlite_url(url):
        columns = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        values = [tuple(row.get(col) for col in columns) for row in rows]
        with _sqlite_conn(url) as conn:
            conn.executemany(sql, values)
            conn.commit()
        return len(rows)

    engine = _pg_engine(url)
    pd.DataFrame(rows).to_sql(table, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return len(rows)


def save_etf_spot_snapshot(df: pd.DataFrame, snapshot_ts: str | None = None) -> dict[str, Any]:
    init_db()
    snapshot_ts = snapshot_ts or now_iso()
    rows = []
    for payload in _json_ready_records(df):
        code = str(payload.get("代码", "")).zfill(6)
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "code": code,
                "name": str(payload.get("名称", "")),
                "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
            }
        )
    return {"table": "etf_spot_snapshot", "snapshot_ts": snapshot_ts, "rows": _insert_rows("etf_spot_snapshot", rows)}


def save_sector_heat_snapshot(df: pd.DataFrame, snapshot_ts: str | None = None) -> dict[str, Any]:
    init_db()
    snapshot_ts = snapshot_ts or now_iso()
    rows = []
    for payload in _json_ready_records(df):
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "sector": str(payload.get("sector") or payload.get("板块") or ""),
                "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
            }
        )
    return {"table": "sector_heat_snapshot", "snapshot_ts": snapshot_ts, "rows": _insert_rows("sector_heat_snapshot", rows)}


def save_score_snapshot(code: str, name: str, model: dict[str, Any], snapshot_ts: str | None = None) -> dict[str, Any]:
    init_db()
    snapshot_ts = snapshot_ts or now_iso()
    payload = {
        "code": code,
        "name": name,
        "total_score": model.get("total_score"),
        "action": model.get("action"),
        "factor_scores": model.get("factor_scores"),
        "raw": model.get("raw"),
        "positives": model.get("positives"),
        "negatives": model.get("negatives"),
    }
    rows = [
        {
            "snapshot_ts": snapshot_ts,
            "code": code,
            "name": name,
            "total_score": model.get("total_score"),
            "action": model.get("action"),
            "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
        }
    ]
    return {"table": "score_snapshot", "snapshot_ts": snapshot_ts, "rows": _insert_rows("score_snapshot", rows)}


def _latest_ts(table: str) -> str | None:
    init_db()
    sql = f"SELECT MAX(snapshot_ts) AS snapshot_ts FROM {table}"
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            row = conn.execute(sql).fetchone()
            return row[0] if row and row[0] else None
    from sqlalchemy import text

    engine = _pg_engine(url)
    with engine.begin() as conn:
        row = conn.execute(text(sql)).fetchone()
        return row[0] if row and row[0] else None


def _load_latest_payloads(table: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    latest = _latest_ts(table)
    if not latest:
        return None, {"ok": False, "label": f"{table}-数据库", "detail": "暂无快照"}

    sql = f"SELECT payload_json FROM {table} WHERE snapshot_ts = ?"
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            rows = conn.execute(sql, (latest,)).fetchall()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(text(sql.replace("?", ":snapshot_ts")), {"snapshot_ts": latest}).fetchall()

    records = [json.loads(row[0]) for row in rows]
    return pd.DataFrame(records), {"ok": True, "label": f"{table}-数据库", "detail": f"{len(records):,} 行，快照 {latest}"}


def load_latest_etf_spot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_latest_payloads("etf_spot_snapshot")


def load_latest_sector_heat() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_latest_payloads("sector_heat_snapshot")


def save_watchlist(codes: list[str], owner: str = "default") -> int:
    init_db()
    clean_codes = []
    for item in codes:
        code = "".join(ch for ch in str(item) if ch.isdigit())[-6:]
        if code and code not in clean_codes:
            clean_codes.append(code.zfill(6))

    url = database_url()
    created_at = now_iso()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            conn.execute("DELETE FROM watchlist WHERE owner = ?", (owner,))
            conn.executemany(
                "INSERT INTO watchlist (owner, code, created_at) VALUES (?, ?, ?)",
                [(owner, code, created_at) for code in clean_codes],
            )
            conn.commit()
        return len(clean_codes)

    from sqlalchemy import text

    engine = _pg_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM watchlist WHERE owner = :owner"), {"owner": owner})
        for code in clean_codes:
            conn.execute(
                text("INSERT INTO watchlist (owner, code, created_at) VALUES (:owner, :code, :created_at)"),
                {"owner": owner, "code": code, "created_at": created_at},
            )
    return len(clean_codes)


def load_watchlist(owner: str = "default") -> list[str]:
    init_db()
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            rows = conn.execute("SELECT code FROM watchlist WHERE owner = ? ORDER BY created_at, code", (owner,)).fetchall()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT code FROM watchlist WHERE owner = :owner ORDER BY created_at, code"), {"owner": owner}).fetchall()
    return [str(row[0]).zfill(6) for row in rows]


def database_summary() -> dict[str, Any]:
    init_db()
    summary: dict[str, Any] = {"url": masked_database_url(), "backend": "SQLite" if is_sqlite_url() else "PostgreSQL"}
    for table in ["etf_spot_snapshot", "sector_heat_snapshot", "score_snapshot", "watchlist"]:
        latest = _latest_ts(table) if table != "watchlist" else None
        url = database_url()
        if is_sqlite_url(url):
            with _sqlite_conn(url) as conn:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        else:
            from sqlalchemy import text

            engine = _pg_engine(url)
            with engine.begin() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
        summary[table] = {"rows": count, "latest": latest}
    return summary
