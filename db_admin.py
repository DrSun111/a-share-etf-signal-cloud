from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import db_store


TABLES = [
    "etf_spot_snapshot",
    "sector_heat_snapshot",
    "score_snapshot",
    "otc_watch_snapshot",
    "watchlist",
]

KEY_COLUMNS = {
    "etf_spot_snapshot": ("snapshot_ts", "code"),
    "sector_heat_snapshot": ("snapshot_ts", "sector"),
    "score_snapshot": ("snapshot_ts", "code"),
    "otc_watch_snapshot": ("snapshot_ts", "code"),
    "watchlist": ("owner", "code"),
}


def normalize_code_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    parts = re.split(r"[\s,;|/]+", raw) if isinstance(raw, str) else list(raw)
    codes: list[str] = []
    for item in parts:
        code = "".join(ch for ch in str(item or "") if ch.isdigit())[-6:]
        if code:
            code = code.zfill(6)
            if code not in codes:
                codes.append(code)
    return codes


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def parse_tables(value: str) -> list[str]:
    if value.lower().strip() == "all":
        return TABLES
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in TABLES]
    if unknown:
        raise SystemExit(f"Unknown table(s): {', '.join(unknown)}")
    return requested


def sqlite_table_rows(source: Path, table: str) -> list[dict[str, Any]]:
    with sqlite3.connect(source) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if not exists:
            return []
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]


def key_for(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple("" if row.get(column) is None else str(row.get(column)) for column in columns)


def existing_target_keys(table: str, columns: tuple[str, ...]) -> set[tuple[str, ...]]:
    db_store.init_db()
    selected = ", ".join(columns)
    sql = f"SELECT {selected} FROM {table}"
    url = db_store.database_url()
    if db_store.is_sqlite_url(url):
        with db_store._sqlite_conn(url) as conn:
            rows = conn.execute(sql).fetchall()
    else:
        from sqlalchemy import text

        engine = db_store._pg_engine(url)
        with engine.begin() as conn:
            rows = conn.execute(text(sql)).fetchall()
    return {tuple("" if value is None else str(value) for value in row) for row in rows}


def insert_rows(table: str, rows: list[dict[str, Any]], chunk_size: int = 1000) -> int:
    inserted = 0
    for index in range(0, len(rows), chunk_size):
        inserted += db_store._insert_rows(table, rows[index : index + chunk_size])
    return inserted


def command_summary(_: argparse.Namespace) -> dict[str, Any]:
    return db_store.database_summary()


def command_init(_: argparse.Namespace) -> dict[str, Any]:
    db_store.init_db()
    return {"ok": True, "database": db_store.masked_database_url(), "summary": db_store.database_summary()}


def command_seed_watchlist(args: argparse.Namespace) -> dict[str, Any]:
    etf_codes = normalize_code_list(args.etf or os.environ.get("ETF_WATCHLIST_CODES"))
    otc_codes = normalize_code_list(args.otc or os.environ.get("OTC_WATCHLIST_CODES"))
    result: dict[str, Any] = {
        "ok": True,
        "database": db_store.masked_database_url(),
        "etf_codes": etf_codes,
        "otc_codes": otc_codes,
    }
    if etf_codes:
        result["etf_saved"] = db_store.save_watchlist(etf_codes, owner="default")
    if otc_codes:
        result["otc_saved"] = db_store.save_watchlist(otc_codes, owner="otc")
    if not etf_codes and not otc_codes:
        result["detail"] = "No ETF_WATCHLIST_CODES or OTC_WATCHLIST_CODES provided."
    return result


def command_migrate_sqlite(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser() if args.source else db_store.LOCAL_DB_PATH
    source = source.resolve()
    if not source.exists():
        raise SystemExit(f"SQLite source does not exist: {source}")

    target_url = db_store.database_url()
    if db_store.is_sqlite_url(target_url):
        target_path = db_store._sqlite_path(target_url).resolve()
        if target_path == source:
            return {
                "ok": False,
                "database": db_store.masked_database_url(),
                "source": str(source),
                "detail": "Target database is the same local SQLite file. Set DATABASE_URL to a cloud PostgreSQL URL first.",
            }

    db_store.init_db()
    table_names = parse_tables(args.tables)
    skip_existing = not args.no_skip_existing
    result: dict[str, Any] = {
        "ok": True,
        "database": db_store.masked_database_url(),
        "source": str(source),
        "skip_existing": skip_existing,
        "tables": {},
    }

    for table in table_names:
        rows = sqlite_table_rows(source, table)
        read_count = len(rows)
        skipped = 0
        if skip_existing and rows:
            key_columns = KEY_COLUMNS[table]
            existing = existing_target_keys(table, key_columns)
            seen: set[tuple[str, ...]] = set()
            filtered: list[dict[str, Any]] = []
            for row in rows:
                key = key_for(row, key_columns)
                if key in existing or key in seen:
                    skipped += 1
                    continue
                seen.add(key)
                filtered.append(row)
            rows = filtered
        inserted = insert_rows(table, rows) if rows else 0
        result["tables"][table] = {"read": read_count, "inserted": inserted, "skipped": skipped}

    result["summary"] = db_store.database_summary()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Database utility for the A-share ETF signal app.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="Print database backend and table counts.")
    summary.set_defaults(func=command_summary)

    init = subparsers.add_parser("init", help="Create database tables and indexes.")
    init.set_defaults(func=command_init)

    seed = subparsers.add_parser("seed-watchlist", help="Save ETF/OTC watchlist codes into the configured database.")
    seed.add_argument("--etf", default="", help="ETF codes separated by commas or spaces. Defaults to ETF_WATCHLIST_CODES.")
    seed.add_argument("--otc", default="", help="OTC fund codes separated by commas or spaces. Defaults to OTC_WATCHLIST_CODES.")
    seed.set_defaults(func=command_seed_watchlist)

    migrate = subparsers.add_parser("migrate-sqlite", help="Append local SQLite snapshots into the configured database.")
    migrate.add_argument("--source", default="", help="SQLite file path. Defaults to data/etf_signal.db.")
    migrate.add_argument("--tables", default="all", help="Comma-separated table names, or all.")
    migrate.add_argument("--no-skip-existing", action="store_true", help="Append all rows, including duplicate snapshot keys.")
    migrate.set_defaults(func=command_migrate_sqlite)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print_json(args.func(args))


if __name__ == "__main__":
    main()
