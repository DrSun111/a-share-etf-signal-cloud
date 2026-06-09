from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
import requests


APP_DIR = Path(__file__).resolve().parent
OTC_WATCHLIST_FILE = APP_DIR / "otc_watchlist.json"
CACHE_DIR = APP_DIR / "data" / "vendor_cache"
DEFAULT_OTC_WATCHLIST = ["110022", "161725", "005827", "001071", "000001"]
WEIGHTS = {"trend": 0.25, "momentum": 0.20, "holding": 0.20, "heat": 0.20, "risk": 0.15}


def maybe_clean_code(value: Any) -> str:
    code = re.sub(r"\D", "", str(value or ""))
    return code[-6:].zfill(6) if code else ""


def normalize_code_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    parts = re.split(r"[\s,，;；/]+", raw) if isinstance(raw, str) else list(raw)
    codes: list[str] = []
    for item in parts:
        code = maybe_clean_code(item)
        if code and code not in codes:
            codes.append(code)
    return codes


def load_otc_watchlist() -> list[str]:
    env_codes = normalize_code_list(os.environ.get("OTC_WATCHLIST_CODES"))
    if env_codes:
        return env_codes
    merged: list[str] = []
    try:
        from db_store import load_watchlist as load_db_watchlist

        db_codes = normalize_code_list(load_db_watchlist(owner="otc"))
        merged.extend(code for code in db_codes if code not in merged)
    except Exception:  # noqa: BLE001
        pass
    try:
        if OTC_WATCHLIST_FILE.exists():
            codes = normalize_code_list(json.loads(OTC_WATCHLIST_FILE.read_text(encoding="utf-8")))
            merged.extend(code for code in codes if code not in merged)
    except (OSError, json.JSONDecodeError):
        pass
    return merged or DEFAULT_OTC_WATCHLIST


def to_num(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def clamp(value: float, lo: float = 0, hi: float = 100) -> float:
    if value is None or pd.isna(value) or math.isinf(value):
        return 50.0
    return float(max(lo, min(hi, value)))


def linear_score(value: float, low: float, high: float) -> float:
    if pd.isna(value) or high == low:
        return 50.0
    return clamp((value - low) / (high - low) * 100)


def percentile_score(value: float, series: pd.Series | None) -> float:
    value = to_num(value)
    if pd.isna(value) or series is None:
        return 50.0
    clean = numeric_series(pd.Series(series)).dropna()
    if clean.empty:
        return 50.0
    return clamp((clean <= value).mean() * 100)


def run_vendor(label: str, fn, *args, **kwargs) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    try:
        df = fn(*args, **kwargs)
        if df is None or df.empty:
            return None, {"label": label, "ok": False, "detail": "返回为空"}
        return df, {"label": label, "ok": True, "detail": f"{len(df):,} 行"}
    except Exception as exc:  # noqa: BLE001 - public endpoints often fail transiently.
        return None, {"label": label, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return CACHE_DIR / f"{safe}.pkl"


def failure_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return CACHE_DIR / f"{safe}.fail"


def load_cached_df(key: str, ttl_seconds: int | None) -> tuple[pd.DataFrame | None, float | None]:
    path = cache_path(key)
    if not path.exists():
        return None, None
    age = time.time() - path.stat().st_mtime
    if ttl_seconds is not None and age > ttl_seconds:
        return None, age
    try:
        return pd.read_pickle(path), age
    except Exception:  # noqa: BLE001
        return None, age


def save_cached_df(key: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path(key))
    except Exception:  # noqa: BLE001 - cache failure must not block collection.
        return


def mark_failure(key: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        failure_path(key).write_text(datetime.now().isoformat(), encoding="utf-8")
    except Exception:  # noqa: BLE001
        return


def recent_failure_age(key: str, cooldown_seconds: int) -> float | None:
    path = failure_path(key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    return age if age < cooldown_seconds else None


def cached_vendor(
    label: str,
    key: str,
    ttl_seconds: int,
    fn,
    *args,
    stale_on_error: bool = True,
    failure_cooldown: int = 0,
    **kwargs,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    cached, age = load_cached_df(key, ttl_seconds)
    if cached is not None:
        return cached.copy(), {"label": label, "ok": True, "detail": f"本地缓存 {len(cached):,} 行，约 {int(age or 0)} 秒前"}

    if failure_cooldown:
        fail_age = recent_failure_age(key, failure_cooldown)
        if fail_age is not None:
            stale, stale_age = load_cached_df(key, None)
            if stale is not None:
                return stale.copy(), {
                    "label": label,
                    "ok": True,
                    "detail": f"近期接口失败，使用过期缓存 {len(stale):,} 行，约 {int(stale_age or 0)} 秒前",
                }
            return None, {"label": label, "ok": False, "detail": f"近期接口失败，{int(fail_age)} 秒前已跳过重试"}

    df, status = run_vendor(label, fn, *args, **kwargs)
    if df is not None:
        save_cached_df(key, df)
        return df, status

    if failure_cooldown:
        mark_failure(key)

    if stale_on_error:
        stale, stale_age = load_cached_df(key, None)
        if stale is not None:
            return stale.copy(), {
                "label": label,
                "ok": True,
                "detail": f"接口失败，使用过期缓存 {len(stale):,} 行，约 {int(stale_age or 0)} 秒前；原错误：{status.get('detail')}",
            }
    return df, status


def get_open_fund_daily() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = cached_vendor("开放式基金净值-东方财富/天天基金", "open_fund_daily", 900, ak.fund_open_fund_daily_em)
    if df is None or "基金代码" not in df.columns:
        return None, status
    out = df.copy()
    out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
    for col in out.columns:
        if "单位净值" in str(col) or "累计净值" in str(col) or col in {"日增长值", "日增长率"}:
            out[col] = numeric_series(out[col])
    return out, status


def get_fund_names() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = cached_vendor("全部基金名称-东方财富/天天基金", "fund_names", 86400, ak.fund_name_em)
    if df is None or "基金代码" not in df.columns:
        return None, status
    out = df.copy()
    out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
    return out, status


def get_open_fund_estimation() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = cached_vendor("场外基金盘中估值-东方财富", "open_fund_estimation", 180, ak.fund_value_estimation_em)
    if df is None or "基金代码" not in df.columns:
        return None, status
    out = df.copy()
    out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
    rename: dict[str, str] = {}
    estimate_date = ""
    for col in out.columns:
        text = str(col)
        if "估算数据-估算值" in text:
            rename[col] = "估算净值"
            match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            estimate_date = match.group(1) if match else estimate_date
        elif "估算数据-估算增长率" in text:
            rename[col] = "估算涨幅"
            match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            estimate_date = match.group(1) if match else estimate_date
        elif "公布数据-单位净值" in text:
            rename[col] = "公布单位净值"
        elif "公布数据-日增长率" in text:
            rename[col] = "公布日增长率"
        elif text.endswith("单位净值") and "公布数据" not in text:
            rename[col] = "上一净值"
    out = out.rename(columns=rename)
    for col in ["估算净值", "估算涨幅", "估算偏差", "公布单位净值", "公布日增长率", "上一净值"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out["估算日期"] = estimate_date or datetime.now().strftime("%Y-%m-%d")
    keep = ["基金代码", "基金名称", "估算净值", "估算涨幅", "估算偏差", "公布单位净值", "公布日增长率", "上一净值", "估算日期"]
    return out[[col for col in keep if col in out.columns]], status


def get_a_spot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = cached_vendor("A股实时行情-东方财富", "a_spot", 90, ak.stock_zh_a_spot_em, failure_cooldown=300)
    if df is None or "代码" not in df.columns:
        return None, status
    out = df.copy()
    out["股票代码"] = out["代码"].astype(str).str.zfill(6)
    for col in ["最新价", "涨跌幅", "成交额", "主力净流入-净额"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    return out, status


def get_sina_holding_quotes(codes: list[str]) -> pd.DataFrame:
    clean_codes = normalize_code_list(codes)[:40]
    if not clean_codes:
        return pd.DataFrame()
    symbols = [("sh" if code.startswith(("5", "6", "9")) else "sz") + code for code in clean_codes]
    try:
        resp = requests.get(
            "https://hq.sinajs.cn/list=" + ",".join(symbols),
            headers={"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        text = resp.content.decode("gbk", errors="ignore")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    pattern = re.compile(r'var hq_str_(?:sh|sz)(\d{6})="([^"]*)";')
    for code, payload in pattern.findall(text):
        parts = payload.split(",")
        if len(parts) < 32 or not parts[0]:
            continue
        prev_close = to_num(parts[2])
        latest = to_num(parts[3])
        pct = (latest / prev_close - 1) * 100 if pd.notna(latest) and pd.notna(prev_close) and prev_close else np.nan
        rows.append(
            {
                "股票代码": code,
                "股票名称": parts[0],
                "最新价": latest,
                "涨跌幅": pct,
                "成交额": to_num(parts[9]),
                "主力净流入-净额": np.nan,
            }
        )
    return pd.DataFrame(rows)


def latest_row(df: pd.DataFrame | None, code: str, code_col: str = "基金代码") -> pd.Series | None:
    if df is None or df.empty or code_col not in df.columns:
        return None
    hit = df[df[code_col].astype(str).str.zfill(6) == maybe_clean_code(code)]
    return hit.iloc[0] if not hit.empty else None


def fetch_open_fund_nav_history_em(code: str, max_pages: int = 14) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages + 1):
        url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page={page}&per=20&sdate=&edate="
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            match = re.search(r'content:"(.*?)",records:', resp.text, flags=re.S)
            if not match:
                break
            fragment = html_lib.unescape(match.group(1).replace('\\"', '"'))
            tables = pd.read_html(StringIO(fragment))
        except Exception:  # noqa: BLE001
            break
        if not tables or tables[0].empty:
            break
        frames.append(tables[0])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "净值日期" in out.columns:
        out["净值日期"] = pd.to_datetime(out["净值日期"], errors="coerce")
        out = out.dropna(subset=["净值日期"]).drop_duplicates("净值日期").sort_values("净值日期")
    for col in out.columns:
        if "单位净值" in str(col) or "累计净值" in str(col) or "日增长率" in str(col):
            out[col] = numeric_series(out[col])
    return out.reset_index(drop=True)


def get_nav_history(code: str) -> pd.DataFrame:
    cached, _ = load_cached_df(f"nav_history_{code}", 900)
    if cached is not None:
        return cached.copy()
    df = fetch_open_fund_nav_history_em(code)
    if not df.empty:
        save_cached_df(f"nav_history_{code}", df)
        return df
    fallback, _ = run_vendor(f"场外基金单位净值走势-{code}", ak.fund_open_fund_info_em, symbol=code, indicator="单位净值走势", period="成立来")
    if fallback is not None:
        save_cached_df(f"nav_history_{code}", fallback)
        return fallback
    stale, _ = load_cached_df(f"nav_history_{code}", None)
    return stale.copy() if stale is not None else pd.DataFrame()


def nav_to_price_df(nav_df: pd.DataFrame | None) -> pd.DataFrame:
    if nav_df is None or nav_df.empty:
        return pd.DataFrame()
    df = nav_df.copy()
    date_col = "净值日期" if "净值日期" in df.columns else df.columns[0]
    nav_col = "单位净值" if "单位净值" in df.columns else None
    if nav_col is None:
        candidates = [col for col in df.columns if col != date_col]
        nav_col = candidates[0] if candidates else None
    if nav_col is None:
        return pd.DataFrame()
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "close": numeric_series(df[nav_col])}).dropna()
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    pct_col = next((col for col in df.columns if "日增长率" in str(col)), None)
    if pct_col:
        pct_map = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "vendor_pct": numeric_series(df[pct_col])})
        out = out.merge(pct_map.dropna(subset=["date"]).drop_duplicates("date"), on="date", how="left")
    else:
        out["vendor_pct"] = np.nan
    out["pct"] = out["vendor_pct"].combine_first(out["close"].pct_change() * 100)
    return out[["date", "close", "pct"]]


def add_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return price_df
    out = price_df.copy().sort_values("date").reset_index(drop=True)
    for window in [5, 10, 20, 60, 120]:
        out[f"ma{window}"] = out["close"].rolling(window, min_periods=max(2, min(window, 5))).mean()
        out[f"high{window}"] = out["close"].rolling(window, min_periods=max(2, min(window, 5))).max()
    out["return5"] = out["close"].pct_change(5) * 100
    out["return10"] = out["close"].pct_change(10) * 100
    out["return20"] = out["close"].pct_change(20) * 100
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["drawdown120"] = (out["close"] / out["high120"] - 1).abs() * 100
    out["volatility20"] = out["pct"].rolling(20, min_periods=5).std()
    return out


def get_holdings(code: str) -> pd.DataFrame:
    cached, _ = load_cached_df(f"holdings_{code}", 21600)
    if cached is not None:
        return cached.copy()
    current_year = date.today().year
    for year in range(current_year, current_year - 4, -1):
        df, _ = run_vendor(f"场外基金股票持仓-{year}", ak.fund_portfolio_hold_em, symbol=code, date=str(year))
        if df is None:
            continue
        out = df.copy()
        for col in ["占净值比例", "持股数", "持仓市值"]:
            if col in out.columns:
                out[col] = numeric_series(out[col])
        save_cached_df(f"holdings_{code}", out)
        return out
    stale, _ = load_cached_df(f"holdings_{code}", None)
    return stale.copy() if stale is not None else pd.DataFrame()


def holding_impact(holdings_df: pd.DataFrame, a_spot: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, float]]:
    metrics = {"holding_contribution": np.nan, "top_weight": np.nan, "up_ratio": np.nan, "up_count": np.nan, "main_flow": np.nan}
    if holdings_df is None or holdings_df.empty or "股票代码" not in holdings_df.columns:
        return pd.DataFrame(), metrics
    top = holdings_df.head(20).copy()
    top["股票代码"] = top["股票代码"].astype(str).str.zfill(6)
    weight_col = "占净值比例" if "占净值比例" in top.columns else next((col for col in top.columns if "比例" in str(col)), None)
    if weight_col:
        top["占净值比例"] = numeric_series(top[weight_col])
        metrics["top_weight"] = float(top["占净值比例"].dropna().sum()) if top["占净值比例"].notna().any() else np.nan
    if a_spot is not None and not a_spot.empty:
        keep_cols = ["股票代码", "最新价", "涨跌幅", "成交额", "主力净流入-净额"]
        top = top.merge(a_spot[[col for col in keep_cols if col in a_spot.columns]], on="股票代码", how="left")
        if "占净值比例" in top.columns and "涨跌幅" in top.columns:
            top["估算贡献"] = top["占净值比例"] * top["涨跌幅"] / 100
            metrics["holding_contribution"] = float(top["估算贡献"].dropna().sum()) if top["估算贡献"].notna().any() else np.nan
        if "涨跌幅" in top.columns:
            pct = numeric_series(top["涨跌幅"]).dropna()
            if not pct.empty:
                metrics["up_ratio"] = float((pct > 0).mean() * 100)
                metrics["up_count"] = float((pct > 0).sum())
        if "主力净流入-净额" in top.columns:
            flow = numeric_series(top["主力净流入-净额"]).dropna()
            metrics["main_flow"] = float(flow.sum()) if not flow.empty else np.nan
    return top, metrics


def score_watch_item(
    code: str,
    daily_row: pd.Series | None,
    name_row: pd.Series | None,
    est_row: pd.Series | None,
    estimation_df: pd.DataFrame | None,
    daily_df: pd.DataFrame | None,
    nav_df: pd.DataFrame,
    holding_metrics: dict[str, float],
) -> dict[str, Any]:
    price_df = nav_to_price_df(nav_df)
    ind = add_indicators(price_df) if not price_df.empty else pd.DataFrame()
    last = ind.iloc[-1] if not ind.empty else pd.Series(dtype="float64")
    close = to_num(last.get("close"))
    ma = {w: to_num(last.get(f"ma{w}")) for w in [5, 10, 20, 60]}
    gt = lambda a, b: pd.notna(a) and pd.notna(b) and a > b
    above_ma = sum(gt(close, ma[w]) for w in [5, 10, 20, 60])
    alignment = sum(gt(ma[a], ma[b]) for a, b in [(5, 10), (10, 20), (20, 60)])
    ret5 = to_num(last.get("return5"))
    ret10 = to_num(last.get("return10"))
    ret20 = to_num(last.get("return20"))
    macd_hist = to_num(last.get("macd_hist"))
    macd = to_num(last.get("macd"))
    dea = to_num(last.get("macd_signal"))
    macd_score = 70 if gt(macd, dea) and macd_hist > 0 else (35 if pd.notna(macd_hist) and macd_hist < 0 else 50)

    trend_score = clamp(
        above_ma / 4 * 30
        + alignment / 3 * 22
        + linear_score(ret5, -4.5, 6.5) * 0.16
        + linear_score(ret20, -9, 13) * 0.17
        + macd_score * 0.15
    )

    estimate_pct = to_num(est_row.get("估算涨幅")) if est_row is not None else np.nan
    daily_pct = to_num(daily_row.get("日增长率")) if daily_row is not None else to_num(last.get("pct"))
    positive_days5 = int((ind["pct"].tail(5) > 0).sum()) if not ind.empty and "pct" in ind.columns else 0

    contribution = to_num(holding_metrics.get("holding_contribution"))
    top_weight = to_num(holding_metrics.get("top_weight"))
    up_ratio = to_num(holding_metrics.get("up_ratio"))
    main_flow = to_num(holding_metrics.get("main_flow"))
    through_pct = contribution if pd.notna(contribution) else np.nan
    realtime_pct = estimate_pct if pd.notna(estimate_pct) else (through_pct if pd.notna(through_pct) else daily_pct)
    momentum_score = clamp(
        linear_score(realtime_pct, -2.5, 2.8) * 0.36
        + linear_score(ret5, -4.5, 6.5) * 0.24
        + linear_score(ret20, -9, 13) * 0.20
        + linear_score(positive_days5, 1, 4) * 0.20
    )
    flow_score = linear_score(main_flow, -8e8, 8e8)
    holding_score = clamp(
        linear_score(contribution, -1.1, 1.2) * 0.44
        + linear_score(up_ratio, 32, 70) * 0.26
        + linear_score(top_weight, 20, 70) * 0.18
        + flow_score * 0.12
    )

    if estimation_df is not None and "估算涨幅" in estimation_df.columns and pd.notna(estimate_pct):
        peer_rank = percentile_score(estimate_pct, estimation_df["估算涨幅"])
    elif daily_df is not None and "日增长率" in daily_df.columns:
        peer_rank = percentile_score(realtime_pct, daily_df["日增长率"])
    else:
        peer_rank = 50.0
    heat_score = clamp(peer_rank * 0.68 + linear_score(realtime_pct, -2.5, 2.8) * 0.32)

    drawdown120 = to_num(last.get("drawdown120"))
    volatility20 = to_num(last.get("volatility20"))
    overheat_score = 100.0
    if pd.notna(ret5) and ret5 > 7:
        overheat_score -= min(30, (ret5 - 7) * 4)
    if pd.notna(ret10) and ret10 > 12:
        overheat_score -= min(24, (ret10 - 12) * 2.5)
    if pd.notna(realtime_pct) and realtime_pct > 3.5:
        overheat_score -= min(18, (realtime_pct - 3.5) * 5)
    risk_score = clamp(
        linear_score(drawdown120, 22, 3) * 0.36
        + linear_score(volatility20, 4.2, 0.6) * 0.22
        + clamp(overheat_score) * 0.25
        + linear_score(top_weight, 85, 30) * 0.17
    )

    total = (
        trend_score * WEIGHTS["trend"]
        + momentum_score * WEIGHTS["momentum"]
        + holding_score * WEIGHTS["holding"]
        + heat_score * WEIGHTS["heat"]
        + risk_score * WEIGHTS["risk"]
    )
    broken_ma20 = pd.notna(close) and pd.notna(ma[20]) and close < ma[20]
    broken_ma60 = pd.notna(close) and pd.notna(ma[60]) and close < ma[60]
    holding_drag = pd.notna(contribution) and contribution < -0.30 and (pd.isna(up_ratio) or up_ratio < 45)
    overheat = pd.notna(ret5) and ret5 > 8 and pd.notna(ret10) and ret10 > 12 and risk_score < 55

    if overheat:
        action = "禁止追高"
    elif broken_ma60 and (heat_score < 48 or holding_score < 45):
        action = "离场"
    elif broken_ma20 or holding_drag or (pd.notna(drawdown120) and drawdown120 > 12):
        action = "减仓"
    elif total >= 72 and trend_score >= 62 and momentum_score >= 55 and holding_score >= 50 and risk_score >= 48:
        action = "买入观察"
    elif total >= 58 and not broken_ma20 and not holding_drag:
        action = "持有"
    else:
        action = "观察等待"

    name = ""
    if daily_row is not None:
        name = str(daily_row.get("基金简称") or "")
    if not name and est_row is not None:
        name = str(est_row.get("基金名称") or "")
    if not name and name_row is not None:
        name = str(name_row.get("基金简称") or "")
    if not name:
        name = code

    fund_type = ""
    if name_row is not None:
        fund_type = str(name_row.get("基金类型") or "")
    if not fund_type and daily_row is not None:
        fund_type = str(daily_row.get("基金类型") or "")

    unit_nav_col = next((col for col in daily_row.index if "单位净值" in str(col)), None) if daily_row is not None else None
    unit_nav = to_num(daily_row.get(unit_nav_col)) if daily_row is not None and unit_nav_col else close
    if pd.isna(unit_nav):
        unit_nav = close
    estimated_nav = to_num(est_row.get("估算净值")) if est_row is not None else np.nan
    through_nav = unit_nav * (1 + through_pct / 100) if pd.notna(unit_nav) and pd.notna(through_pct) else np.nan
    if pd.isna(estimated_nav) and pd.notna(through_nav):
        estimated_nav = through_nav
    realtime_through_pct = through_pct

    return {
        "基金代码": code,
        "基金简称": name,
        "基金类型": fund_type,
        "最新单位净值": unit_nav,
        "净值日期": str(pd.to_datetime(last.get("date")).date()) if pd.notna(last.get("date")) else "",
        "日增长率": daily_pct,
        "估算净值": estimated_nav,
        "估算涨幅": estimate_pct,
        "估算偏差": to_num(est_row.get("估算偏差")) if est_row is not None else np.nan,
        "实时穿透净值": through_nav,
        "实时穿透涨幅": realtime_through_pct,
        "重仓估算贡献": contribution,
        "前20持仓权重": top_weight,
        "上涨重仓股数": holding_metrics.get("up_count"),
        "上涨重仓股比例": up_ratio,
        "近5日": ret5,
        "近10日": ret10,
        "近20日": ret20,
        "120日回撤": drawdown120,
        "DIF": macd,
        "DEA": dea,
        "MACD柱": macd_hist,
        "趋势分": trend_score,
        "净值动能分": momentum_score,
        "持仓穿透分": holding_score,
        "同类热度分": heat_score,
        "风险控制分": risk_score,
        "场外短线评分": clamp(total),
        "动作": action,
        "申购状态": str(daily_row.get("申购状态") or "-") if daily_row is not None else "-",
        "赎回状态": str(daily_row.get("赎回状态") or "-") if daily_row is not None else "-",
        "手续费": str(daily_row.get("手续费") or "-") if daily_row is not None else "-",
        "快照时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据来源": "东财估值 + 净值历史 + 公开重仓股实时穿透",
    }


def collect_otc_watch_snapshot(codes: list[str] | str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    clean_codes = normalize_code_list(codes) if codes is not None else load_otc_watchlist()
    if not clean_codes:
        clean_codes = DEFAULT_OTC_WATCHLIST

    daily_df, daily_status = get_open_fund_daily()
    names_df, names_status = get_fund_names()
    estimation_df, estimation_status = get_open_fund_estimation()
    a_spot, a_spot_status = get_a_spot()

    errors: list[str] = []
    rows_by_code: dict[str, dict[str, Any]] = {}

    def collect_one(code: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            daily_row = latest_row(daily_df, code)
            name_row = latest_row(names_df, code)
            est_row = latest_row(estimation_df, code)
            nav_df = get_nav_history(code)
            holdings_df = get_holdings(code)
            fund_spot = a_spot
            if (fund_spot is None or fund_spot.empty) and holdings_df is not None and not holdings_df.empty and "股票代码" in holdings_df.columns:
                fund_spot = get_sina_holding_quotes(holdings_df["股票代码"].astype(str).head(20).tolist())
            _, metrics = holding_impact(holdings_df, fund_spot)
            return code, score_watch_item(code, daily_row, name_row, est_row, estimation_df, daily_df, nav_df, metrics), None
        except Exception as exc:  # noqa: BLE001 - keep other watchlist rows usable.
            return (
                code,
                {
                    "基金代码": code,
                    "基金简称": code,
                    "场外短线评分": 50.0,
                    "动作": "数据不足",
                    "快照时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "数据来源": f"采集失败：{type(exc).__name__}: {exc}",
                },
                f"{code}: {type(exc).__name__}: {exc}",
            )

    max_workers = max(1, min(6, len(clean_codes)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect_one, code): code for code in clean_codes}
        for future in as_completed(futures):
            code, row, error = future.result()
            if row is not None:
                rows_by_code[code] = row
            if error:
                errors.append(error)

    rows = [rows_by_code[code] for code in clean_codes if code in rows_by_code]
    out = pd.DataFrame(rows)
    if not out.empty and "场外短线评分" in out.columns:
        out = out.sort_values("场外短线评分", ascending=False, na_position="last").reset_index(drop=True)
    from db_store import save_otc_watch_snapshot

    result = save_otc_watch_snapshot(out)
    result.update(
        {
            "codes": clean_codes,
            "rows": len(out),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "errors": errors,
            "sources": [daily_status, names_status, estimation_status, a_spot_status],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OTC fund watchlist snapshot into the local/cloud database.")
    parser.add_argument("--codes", default="", help="Optional comma/space separated fund codes. Defaults to otc_watchlist.json.")
    args = parser.parse_args()
    codes = normalize_code_list(args.codes) if args.codes else None
    print(json.dumps(collect_otc_watch_snapshot(codes), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
