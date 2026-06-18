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
_INITIALIZED_URLS: set[str] = set()


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
    url = database_url()
    if url in _INITIALIZED_URLS:
        return
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
        CREATE TABLE IF NOT EXISTS otc_watch_snapshot (
            snapshot_ts TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            total_score REAL,
            action TEXT,
            payload_json TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS otc_nav_history (
            code TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
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
        "CREATE INDEX IF NOT EXISTS idx_etf_spot_snapshot_ts ON etf_spot_snapshot (snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_etf_spot_snapshot_code_ts ON etf_spot_snapshot (code, snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_sector_heat_snapshot_ts ON sector_heat_snapshot (snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_score_snapshot_code_ts ON score_snapshot (code, snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_otc_watch_snapshot_ts ON otc_watch_snapshot (snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_otc_watch_snapshot_code_ts ON otc_watch_snapshot (code, snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_otc_nav_history_code_ts ON otc_nav_history (code, snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_watchlist_owner_code ON watchlist (owner, code)",
    ]
    for statement in statements:
        _execute(statement)
    _INITIALIZED_URLS.add(url)


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


def save_otc_watch_snapshot(df: pd.DataFrame, snapshot_ts: str | None = None) -> dict[str, Any]:
    init_db()
    snapshot_ts = snapshot_ts or now_iso()
    rows = []
    for payload in _json_ready_records(df):
        code = str(payload.get("基金代码") or payload.get("code") or "").zfill(6)
        name = str(payload.get("基金简称") or payload.get("基金名称") or payload.get("name") or "")
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "code": code,
                "name": name,
                "total_score": payload.get("场外短线评分") or payload.get("total_score"),
                "action": payload.get("动作") or payload.get("action"),
                "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
            }
        )
    return {"table": "otc_watch_snapshot", "snapshot_ts": snapshot_ts, "rows": _insert_rows("otc_watch_snapshot", rows)}


def save_otc_nav_history(code: str, df: pd.DataFrame | None, snapshot_ts: str | None = None) -> dict[str, Any]:
    init_db()
    code = "".join(ch for ch in str(code or "") if ch.isdigit())[-6:].zfill(6)
    if not code or df is None or df.empty:
        return {"table": "otc_nav_history", "snapshot_ts": snapshot_ts or now_iso(), "rows": 0, "records": 0}

    snapshot_ts = snapshot_ts or now_iso()
    records = _json_ready_records(df)
    row = {
        "code": code,
        "snapshot_ts": snapshot_ts,
        "payload_json": json.dumps(records, ensure_ascii=False, default=str),
    }
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            conn.execute("DELETE FROM otc_nav_history WHERE code = ?", (code,))
            conn.execute(
                "INSERT INTO otc_nav_history (code, snapshot_ts, payload_json) VALUES (?, ?, ?)",
                (row["code"], row["snapshot_ts"], row["payload_json"]),
            )
            conn.commit()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM otc_nav_history WHERE code = :code"), {"code": code})
            conn.execute(
                text(
                    "INSERT INTO otc_nav_history (code, snapshot_ts, payload_json) "
                    "VALUES (:code, :snapshot_ts, :payload_json)"
                ),
                row,
            )
    return {"table": "otc_nav_history", "snapshot_ts": snapshot_ts, "rows": 1, "records": len(records)}


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


def _load_latest_payloads_by_code(table: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    init_db()
    latest = _latest_ts(table)
    if not latest:
        return None, {"ok": False, "label": f"{table}-数据库", "detail": "暂无快照"}

    sql = f"""
        SELECT t.payload_json
        FROM {table} t
        JOIN (
            SELECT code, MAX(snapshot_ts) AS snapshot_ts
            FROM {table}
            GROUP BY code
        ) latest
          ON t.code = latest.code AND t.snapshot_ts = latest.snapshot_ts
        ORDER BY t.snapshot_ts DESC, t.code
    """
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            rows = conn.execute(sql).fetchall()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(text(sql)).fetchall()

    records = [json.loads(row[0]) for row in rows]
    return pd.DataFrame(records), {"ok": True, "label": f"{table}-数据库", "detail": f"{len(records):,} 只基金，每只取最新快照；最新 {latest}"}


def load_latest_etf_spot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_latest_payloads("etf_spot_snapshot")


def load_latest_sector_heat() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_latest_payloads("sector_heat_snapshot")


def load_latest_otc_watch_snapshot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_latest_payloads_by_code("otc_watch_snapshot")


def _load_payload_history_by_code(table: str, code: str, limit: int = 40) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    init_db()
    code = "".join(ch for ch in str(code or "") if ch.isdigit())[-6:].zfill(6)
    if not code:
        return None, {"ok": False, "label": f"{table}-历史快照", "detail": "代码为空"}
    limit = max(1, min(int(limit or 40), 300))
    sql = f"""
        SELECT snapshot_ts, payload_json
        FROM {table}
        WHERE code = ?
        ORDER BY snapshot_ts DESC
        LIMIT ?
    """
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            rows = conn.execute(sql, (code, limit)).fetchall()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(
                text(sql.replace("?", ":code", 1).replace("?", ":limit", 1)),
                {"code": code, "limit": limit},
            ).fetchall()

    records = []
    for snapshot_ts, payload_json in rows:
        payload = json.loads(payload_json)
        payload["_snapshot_ts"] = snapshot_ts
        records.append(payload)
    records.reverse()
    if not records:
        return None, {"ok": False, "label": f"{table}-历史快照", "detail": f"{code} 暂无历史快照"}
    return pd.DataFrame(records), {"ok": True, "label": f"{table}-历史快照", "detail": f"{code} {len(records)} 条"}


def load_etf_spot_history(code: str, limit: int = 40) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_payload_history_by_code("etf_spot_snapshot", code, limit)


def load_otc_watch_history(code: str, limit: int = 40) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return _load_payload_history_by_code("otc_watch_snapshot", code, limit)


def load_sector_heat_history(sector: str | None, limit: int = 40) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    init_db()
    sector = str(sector or "").strip()
    if not sector or sector in {"无板块数据", "未匹配具体行业", "后台快照极速模式", "后台快照缺失"}:
        return None, {"ok": False, "label": "sector_heat_snapshot-历史快照", "detail": "板块为空或未匹配"}
    limit = max(1, min(int(limit or 40), 300))
    sql = """
        SELECT snapshot_ts, payload_json
        FROM sector_heat_snapshot
        WHERE sector = ?
        ORDER BY snapshot_ts DESC
        LIMIT ?
    """
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            rows = conn.execute(sql, (sector, limit)).fetchall()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(text(sql.replace("?", ":sector", 1).replace("?", ":limit", 1)), {"sector": sector, "limit": limit}).fetchall()

    records = []
    for snapshot_ts, payload_json in rows:
        payload = json.loads(payload_json)
        payload["_snapshot_ts"] = snapshot_ts
        records.append(payload)
    records.reverse()
    if not records:
        return None, {"ok": False, "label": "sector_heat_snapshot-历史快照", "detail": f"{sector} 暂无历史快照"}
    return pd.DataFrame(records), {"ok": True, "label": "sector_heat_snapshot-历史快照", "detail": f"{sector} {len(records)} 条"}


def load_latest_otc_nav_history(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    init_db()
    code = "".join(ch for ch in str(code or "") if ch.isdigit())[-6:].zfill(6)
    if not code:
        return None, {"ok": False, "label": "场外基金净值历史-数据库", "detail": "代码为空"}

    sql = "SELECT payload_json, snapshot_ts FROM otc_nav_history WHERE code = ? ORDER BY snapshot_ts DESC LIMIT 1"
    url = database_url()
    if is_sqlite_url(url):
        with _sqlite_conn(url) as conn:
            row = conn.execute(sql, (code,)).fetchone()
    else:
        from sqlalchemy import text

        engine = _pg_engine(url)
        with engine.begin() as conn:
            row = conn.execute(
                text(sql.replace("?", ":code")),
                {"code": code},
            ).fetchone()

    if not row:
        return None, {"ok": False, "label": "场外基金净值历史-数据库", "detail": f"{code} 暂无入库净值历史"}
    records = json.loads(row[0])
    df = pd.DataFrame(records)
    if df.empty:
        return None, {"ok": False, "label": "场外基金净值历史-数据库", "detail": f"{code} 入库净值历史为空"}
    return df, {"ok": True, "label": "场外基金净值历史-数据库", "detail": f"{code} {len(df):,} 行，快照 {row[1]}"}


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
    for table in ["etf_spot_snapshot", "sector_heat_snapshot", "score_snapshot", "otc_watch_snapshot", "otc_nav_history", "watchlist"]:
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
