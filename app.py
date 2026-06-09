from __future__ import annotations

import json
import math
import re
import html
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from collector import collect_market_snapshot
from otc_collector import collect_otc_watch_snapshot
from db_store import (
    database_summary,
    load_latest_etf_spot,
    load_latest_otc_watch_snapshot,
    load_latest_sector_heat,
    save_score_snapshot,
)
from db_store import load_watchlist as load_db_watchlist
from db_store import save_watchlist as save_db_watchlist


st.set_page_config(
    page_title="A股 ETF 短线机会评分台",
    page_icon="ETF",
    layout="wide",
    initial_sidebar_state="expanded",
)


WEIGHTS = {
    "trend": 0.25,
    "volume": 0.20,
    "funds": 0.20,
    "sector": 0.20,
    "risk": 0.15,
}

APP_DIR = Path(__file__).resolve().parent
WATCHLIST_FILE = APP_DIR / "watchlist.json"
OTC_WATCHLIST_FILE = APP_DIR / "otc_watchlist.json"
DEFAULT_WATCHLIST = ["510300", "159915", "512000", "588000", "512880", "159949"]
DEFAULT_OTC_WATCHLIST = ["110022", "161725", "005827", "001071", "000001"]

INDEX_OPTIONS = {
    "沪深300 sh000300": "sh000300",
    "创业板指 sz399006": "sz399006",
    "上证指数 sh000001": "sh000001",
    "深证成指 sz399001": "sz399001",
    "科创50 sh000688": "sh000688",
    "中证500 sh000905": "sh000905",
    "中证1000 sh000852": "sh000852",
}

THEME_TO_SECTOR = [
    (r"证券|券商|证保|金融科技", "证券"),
    (r"银行", "银行"),
    (r"保险", "保险"),
    (r"白酒|酒", "白酒"),
    (r"煤炭", "煤炭开采加工"),
    (r"钢铁", "钢铁"),
    (r"有色|金属|稀土", "金属新材料"),
    (r"芯片|半导体|集成电路", "半导体"),
    (r"机器人|自动化|智能制造|工业母机", "自动化设备"),
    (r"游戏|传媒|影视", "游戏"),
    (r"医药|医疗|创新药|生物", "化学制药"),
    (r"新能源车|汽车|智能车|车联网", "汽车整车"),
    (r"光伏|太阳能", "光伏设备"),
    (r"电池|锂", "电池"),
    (r"军工|国防", "军工电子"),
    (r"电力|公用", "电力"),
    (r"家电", "白色家电"),
    (r"房地产|地产", "房地产开发"),
    (r"农业|养殖|畜牧", "养殖业"),
]

MODEL_REFERENCES = [
    {
        "方向": "趋势/均线/突破",
        "依据": "Brock, Lakonishok & LeBaron (1992), Journal of Finance：检验移动均线和交易区间突破等技术规则。",
        "链接": "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x",
    },
    {
        "方向": "横截面动量",
        "依据": "Jegadeesh & Titman (1993), Journal of Finance：过去表现强弱对后续收益具有统计解释力。",
        "链接": "https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x",
    },
    {
        "方向": "时间序列动量",
        "依据": "Moskowitz, Ooi & Pedersen (2012), Journal of Financial Economics：资产自身过去收益对后续收益有预测信息。",
        "链接": "https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf",
    },
    {
        "方向": "量价确认",
        "依据": "Lee & Swaminathan (2000), Journal of Finance：成交量影响价格动量的强度和持续性。",
        "链接": "https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00280",
    },
    {
        "方向": "流动性/成交额风险",
        "依据": "Amihud (2002), Journal of Financial Markets：低流动性与资产预期收益和价格折价相关。",
        "链接": "https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf",
    },
    {
        "方向": "基金披露边界",
        "依据": "中国证监会《公开募集证券投资基金信息披露管理办法》：公开基金数据以净值、定期报告和组合披露为边界。",
        "链接": "https://www.csrc.gov.cn/csrc/c101877/c1029542/content.shtml",
    },
]


def data_status(label: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"label": label, "ok": bool(ok), "detail": str(detail)}


def maybe_clean_code(value: str) -> str:
    code = re.sub(r"\D", "", value or "")
    return code[-6:].zfill(6) if code else ""


def clean_code(value: str) -> str:
    return maybe_clean_code(value) or "510300"


def extract_code_from_label(label: str) -> str:
    match = re.search(r"\b(\d{6})\b", label or "")
    return match.group(1) if match else ""


def normalize_code_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[\s,，;；|/]+", raw)
    else:
        parts = list(raw)
    codes: list[str] = []
    for item in parts:
        code = maybe_clean_code(str(item))
        if code and code not in codes:
            codes.append(code)
    return codes


def load_watchlist() -> list[str]:
    if st.session_state.get("use_db_watchlist"):
        try:
            codes = normalize_code_list(load_db_watchlist())
            st.session_state["watchlist_codes"] = codes
            return codes
        except Exception as exc:  # noqa: BLE001 - fallback keeps the app usable.
            st.session_state["db_watchlist_error"] = f"{type(exc).__name__}: {exc}"

    if "watchlist_codes" in st.session_state:
        return normalize_code_list(st.session_state["watchlist_codes"])
    codes = DEFAULT_WATCHLIST
    has_saved_file = False
    try:
        if WATCHLIST_FILE.exists():
            has_saved_file = True
            saved = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
            codes = normalize_code_list(saved)
    except (OSError, json.JSONDecodeError):
        codes = DEFAULT_WATCHLIST
    if not codes and not has_saved_file:
        codes = DEFAULT_WATCHLIST
    st.session_state["watchlist_codes"] = codes
    return codes


def save_watchlist(codes: Any) -> list[str]:
    normalized = normalize_code_list(codes)
    st.session_state["watchlist_codes"] = normalized
    if st.session_state.get("use_db_watchlist"):
        try:
            save_db_watchlist(normalized)
        except Exception as exc:  # noqa: BLE001
            st.session_state["db_watchlist_error"] = f"{type(exc).__name__}: {exc}"
        return normalized

    try:
        WATCHLIST_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return normalized


def load_otc_watchlist() -> list[str]:
    if st.session_state.get("use_db_watchlist"):
        try:
            codes = normalize_code_list(load_db_watchlist(owner="otc"))
            st.session_state["otc_watchlist_codes"] = codes
            return codes
        except Exception as exc:  # noqa: BLE001 - fallback keeps the app usable.
            st.session_state["db_watchlist_error"] = f"{type(exc).__name__}: {exc}"

    if "otc_watchlist_codes" in st.session_state:
        return normalize_code_list(st.session_state["otc_watchlist_codes"])
    codes = DEFAULT_OTC_WATCHLIST
    has_saved_file = False
    try:
        if OTC_WATCHLIST_FILE.exists():
            has_saved_file = True
            saved = json.loads(OTC_WATCHLIST_FILE.read_text(encoding="utf-8"))
            codes = normalize_code_list(saved)
    except (OSError, json.JSONDecodeError):
        codes = DEFAULT_OTC_WATCHLIST
    if not codes and not has_saved_file:
        codes = DEFAULT_OTC_WATCHLIST
    st.session_state["otc_watchlist_codes"] = codes
    return codes


def save_otc_watchlist(codes: Any) -> list[str]:
    normalized = normalize_code_list(codes)
    st.session_state["otc_watchlist_codes"] = normalized
    if st.session_state.get("use_db_watchlist"):
        try:
            save_db_watchlist(normalized, owner="otc")
        except Exception as exc:  # noqa: BLE001
            st.session_state["db_watchlist_error"] = f"{type(exc).__name__}: {exc}"
        return normalized

    try:
        OTC_WATCHLIST_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return normalized


def market_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


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


def linear_score(value: float, low: float, high: float, reverse: bool = False) -> float:
    if pd.isna(value):
        return 50.0
    score = (value - low) / (high - low) * 100
    if reverse:
        score = 100 - score
    return clamp(score)


def soft_ratio_score(value: float, ideal_low: float, ideal_high: float, hard_low: float, hard_high: float) -> float:
    if pd.isna(value):
        return 50.0
    if ideal_low <= value <= ideal_high:
        return 88.0
    if value < ideal_low:
        return linear_score(value, hard_low, ideal_low) * 0.8
    return max(35.0, 100 - linear_score(value, ideal_high, hard_high) * 0.65)


def percentile_score(value: float, series: pd.Series, higher_is_better: bool = True) -> float:
    clean = numeric_series(series).dropna()
    if clean.empty or pd.isna(value):
        return 50.0
    rank = (clean <= value).mean() * 100
    return clamp(rank if higher_is_better else 100 - rank)


def format_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1e8:
        return f"{sign}{value / 1e8:.2f}亿"
    if value >= 1e4:
        return f"{sign}{value / 1e4:.2f}万"
    return f"{sign}{value:.0f}"


def pct_text(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}%"


def run_call(label: str, fn, *args, **kwargs) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    try:
        df = fn(*args, **kwargs)
        if df is None or df.empty:
            return None, data_status(label, False, "返回为空")
        return df, data_status(label, True, f"{len(df):,} 行")
    except Exception as exc:  # noqa: BLE001 - data vendors often fail with transient network errors.
        return None, data_status(label, False, f"{type(exc).__name__}: {exc}")


@st.cache_data(ttl=60, show_spinner=False)
def get_etf_spot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return run_call("ETF实时行情-东方财富", ak.fund_etf_spot_em)


@st.cache_data(ttl=900, show_spinner=False)
def get_open_fund_daily() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call("开放式基金净值-东方财富/天天基金", ak.fund_open_fund_daily_em)
    if df is None:
        return None, status
    out = df.copy()
    for col in ["日增长值", "日增长率"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    nav_cols = [col for col in out.columns if "单位净值" in col or "累计净值" in col]
    for col in nav_cols:
        out[col] = numeric_series(out[col])
    return out, status


@st.cache_data(ttl=3600, show_spinner=False)
def get_fund_names() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return run_call("全部基金名称-东方财富/天天基金", ak.fund_name_em)


@st.cache_data(ttl=300, show_spinner=False)
def get_fund_names_from_collector_cache() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    path = APP_DIR / "data" / "vendor_cache" / "fund_names.pkl"
    if not path.exists():
        return None, data_status("全部基金名称-本地后台缓存", False, "暂无本地缓存")
    try:
        df = pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001
        return None, data_status("全部基金名称-本地后台缓存", False, f"{type(exc).__name__}: {exc}")
    if df is None or df.empty:
        return None, data_status("全部基金名称-本地后台缓存", False, "缓存为空")
    out = df.copy()
    if "基金代码" in out.columns:
        out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return out, data_status("全部基金名称-本地后台缓存", True, f"{len(out):,} 行，约 {int(age.total_seconds())} 秒前")


@st.cache_data(ttl=120, show_spinner=False)
def get_open_fund_estimation() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call("场外基金盘中估值-东方财富", ak.fund_value_estimation_em)
    if df is None:
        return None, status
    out = df.copy()
    if "基金代码" not in out.columns:
        return None, data_status("场外基金盘中估值-东方财富", False, "缺少基金代码字段")
    out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)

    rename: dict[str, str] = {}
    estimate_date = ""
    for col in out.columns:
        text = str(col)
        if "估算数据-估算值" in text:
            rename[col] = "估算净值"
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            estimate_date = date_match.group(1) if date_match else estimate_date
        elif "估算数据-估算增长率" in text:
            rename[col] = "估算涨幅"
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
            estimate_date = date_match.group(1) if date_match else estimate_date
        elif "公布数据-单位净值" in text:
            rename[col] = "公布单位净值"
        elif "公布数据-日增长率" in text:
            rename[col] = "公布日增长率"
        elif text.endswith("单位净值") and "公布数据" not in text:
            rename[col] = "上一净值"
    out = out.rename(columns=rename)
    for col in ["估算净值", "估算涨幅", "公布单位净值", "公布日增长率", "估算偏差", "上一净值"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out["估算日期"] = estimate_date or datetime.now().strftime("%Y-%m-%d")
    keep = ["基金代码", "基金名称", "估算净值", "估算涨幅", "估算偏差", "公布单位净值", "公布日增长率", "上一净值", "估算日期"]
    return out[[col for col in keep if col in out.columns]], status


def latest_open_fund_estimation_row(estimation_df: pd.DataFrame | None, code: str) -> pd.Series | None:
    if estimation_df is None or estimation_df.empty or "基金代码" not in estimation_df.columns:
        return None
    hit = estimation_df[estimation_df["基金代码"].astype(str).str.zfill(6) == clean_code(code)]
    return hit.iloc[0] if not hit.empty else None


def fetch_open_fund_nav_history_em(code: str, max_pages: int = 35) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for page in range(1, max_pages + 1):
        url = f"https://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={code}&page={page}&per=20&sdate=&edate="
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        match = re.search(r'content:"(.*?)",records:', resp.text, flags=re.S)
        if not match:
            break
        content = html.unescape(match.group(1).replace(r"\/", "/").replace(r"\"", '"'))
        tables = pd.read_html(StringIO(content))
        if not tables:
            break
        page_df = tables[0]
        if page_df.empty:
            break
        frames.append(page_df)
        pages_match = re.search(r"pages:(\d+)", resp.text)
        if pages_match and page >= int(pages_match.group(1)):
            break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    keep = [col for col in ["净值日期", "单位净值", "累计净值", "日增长率"] if col in df.columns]
    df = df[keep].copy()
    df["净值日期"] = pd.to_datetime(df["净值日期"], errors="coerce")
    for col in ["单位净值", "累计净值", "日增长率"]:
        if col in df.columns:
            df[col] = numeric_series(df[col])
    return df.dropna(subset=["净值日期"]).sort_values("净值日期").reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def get_open_fund_nav_trend(code: str, indicator: str = "单位净值走势") -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call(f"场外基金{indicator}", ak.fund_open_fund_info_em, symbol=code, indicator=indicator, period="成立来")
    if df is None:
        try:
            fallback = fetch_open_fund_nav_history_em(code)
            if not fallback.empty:
                return fallback, data_status("场外基金历史净值-东方财富F10兜底", True, f"{len(fallback):,} 行")
        except Exception as exc:  # noqa: BLE001 - fallback is best effort.
            status = data_status(status["label"], False, f"{status['detail']}；兜底失败 {type(exc).__name__}: {exc}")
        return None, status
    out = df.copy()
    date_col = "净值日期" if "净值日期" in out.columns else out.columns[0]
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    for col in out.columns:
        if col != date_col:
            out[col] = numeric_series(out[col])
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if out.empty:
        try:
            fallback = fetch_open_fund_nav_history_em(code)
            if not fallback.empty:
                return fallback, data_status("场外基金历史净值-东方财富F10兜底", True, f"{len(fallback):,} 行")
        except Exception as exc:  # noqa: BLE001
            return None, data_status(status["label"], False, f"返回为空；兜底失败 {type(exc).__name__}: {exc}")
    return out, status


@st.cache_data(ttl=300, show_spinner=False)
def get_open_fund_nav_from_collector_cache(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    code = clean_code(code)
    path = APP_DIR / "data" / "vendor_cache" / f"nav_history_{code}.pkl"
    if not path.exists():
        return None, data_status("场外基金净值-本地后台缓存", False, f"{code} 暂无本地缓存")
    try:
        df = pd.read_pickle(path)
    except Exception as exc:  # noqa: BLE001
        return None, data_status("场外基金净值-本地后台缓存", False, f"{type(exc).__name__}: {exc}")
    if df is None or df.empty:
        return None, data_status("场外基金净值-本地后台缓存", False, "缓存为空")
    out = df.copy()
    for col in out.columns:
        if "单位净值" in str(col) or "累计净值" in str(col) or "日增长率" in str(col):
            out[col] = numeric_series(out[col])
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return out, data_status("场外基金净值-本地后台缓存", True, f"{len(out):,} 行，约 {int(age.total_seconds())} 秒前")



@st.cache_data(ttl=3600, show_spinner=False)
def get_open_fund_basic(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return run_call("场外基金基本信息-雪球", ak.fund_individual_basic_info_xq, symbol=code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_open_fund_achievement(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call("场外基金业绩-雪球", ak.fund_individual_achievement_xq, symbol=code)
    if df is None:
        return None, status
    out = df.copy()
    for col in ["本产品区间收益", "本产品最大回撒"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    return out, status


@st.cache_data(ttl=3600, show_spinner=False)
def get_open_fund_asset_allocation(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call("场外基金资产配置-雪球", ak.fund_individual_detail_hold_xq, symbol=code)
    if df is None:
        return None, status
    out = df.copy()
    if "仓位占比" in out.columns:
        out["仓位占比"] = numeric_series(out["仓位占比"])
    return out, status


@st.cache_data(ttl=3600, show_spinner=False)
def get_open_fund_holdings(code: str) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    statuses: list[dict[str, Any]] = []
    current_year = date.today().year
    for year in range(current_year, current_year - 4, -1):
        df, status = run_call(f"场外基金股票持仓-{year}", ak.fund_portfolio_hold_em, symbol=code, date=str(year))
        statuses.append(status)
        if df is not None:
            out = df.copy()
            for col in ["占净值比例", "持股数", "持仓市值"]:
                if col in out.columns:
                    out[col] = numeric_series(out[col])
            return out, statuses
    return None, statuses


@st.cache_data(ttl=900, show_spinner=False)
def get_etf_daily(code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    statuses: list[dict[str, Any]] = []
    df, status = run_call(
        "ETF日K-东方财富",
        ak.fund_etf_hist_em,
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    statuses.append(status)

    if df is None:
        df, status = run_call("ETF日K-新浪", ak.fund_etf_hist_sina, symbol=market_symbol(code))
        statuses.append(status)

    if df is None:
        return None, statuses

    out = df.copy()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    out = out.rename(columns=rename_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        statuses.append(data_status("ETF日K字段校验", False, f"缺少字段: {missing}"))
        return None, statuses

    for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out["date"] = pd.to_datetime(out["date"])
    if "amount" not in out.columns or out["amount"].isna().all():
        out["amount"] = out["volume"] * out["close"]
    out = out.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    out = out[(out["date"] >= start_ts) & (out["date"] <= end_ts)].copy()
    return out, statuses


@st.cache_data(ttl=900, show_spinner=False)
def get_index_daily(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    statuses: list[dict[str, Any]] = []
    df, status = run_call(
        "指数日K-东方财富",
        ak.stock_zh_index_daily_em,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    statuses.append(status)
    if df is None:
        df, status = run_call("指数日K-新浪", ak.stock_zh_index_daily, symbol=symbol)
        statuses.append(status)
    if df is None:
        return None, statuses
    out = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
    ).copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)
    out = out[(out["date"] >= pd.to_datetime(start_date)) & (out["date"] <= pd.to_datetime(end_date))].copy()
    return out, statuses


@st.cache_data(ttl=3600, show_spinner=False)
def get_fund_overview(code: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return run_call("基金概况-东方财富", ak.fund_overview_em, symbol=code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fund_nav(code: str, start_date: str, end_date: str) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call(
        "ETF净值-东方财富",
        ak.fund_etf_fund_info_em,
        fund=code,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None:
        return None, status
    out = df.rename(columns={"净值日期": "date", "单位净值": "nav", "日增长率": "nav_pct"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["nav", "nav_pct"]:
        out[col] = numeric_series(out[col])
    return out.sort_values("date").reset_index(drop=True), status


@st.cache_data(ttl=3600, show_spinner=False)
def get_holdings(code: str) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    statuses: list[dict[str, Any]] = []
    current_year = date.today().year
    for year in range(current_year, current_year - 4, -1):
        df, status = run_call(f"成分持仓-{year}", ak.fund_portfolio_hold_em, symbol=code, date=str(year))
        statuses.append(status)
        if df is not None:
            out = df.copy()
            for col in ["占净值比例", "持股数", "持仓市值"]:
                if col in out.columns:
                    out[col] = numeric_series(out[col])
            return out, statuses
    return None, statuses


@st.cache_data(ttl=180, show_spinner=False)
def get_sector_summary() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = run_call("行业热度-同花顺", ak.stock_board_industry_summary_ths)
    if df is None:
        return None, status
    out = df.copy()
    for col in ["涨跌幅", "总成交量", "总成交额", "净流入", "上涨家数", "下跌家数", "领涨股-涨跌幅"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out = out.rename(
        columns={
            "板块": "sector",
            "涨跌幅": "pct_chg",
            "总成交额": "amount",
            "净流入": "net_inflow",
            "上涨家数": "up_count",
            "下跌家数": "down_count",
            "领涨股": "leader",
            "领涨股-涨跌幅": "leader_pct",
        }
    )
    out["up_ratio"] = out["up_count"] / (out["up_count"] + out["down_count"]).replace(0, np.nan) * 100
    out["rank_pct"] = out["pct_chg"].rank(ascending=False, method="min")
    out["rank_amount"] = out["amount"].rank(ascending=False, method="min")
    out["rank_flow"] = out["net_inflow"].rank(ascending=False, method="min")
    return out, status


@st.cache_data(ttl=180, show_spinner=False)
def get_a_spot() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return run_call("A股实时行情-东方财富", ak.stock_zh_a_spot_em)


@st.cache_data(ttl=60, show_spinner=False)
def get_etf_spot_from_db() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return load_latest_etf_spot()


@st.cache_data(ttl=180, show_spinner=False)
def get_sector_summary_from_db() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    df, status = load_latest_sector_heat()
    if df is None or df.empty:
        return None, status
    out = df.copy()
    for col in ["涨跌幅", "总成交量", "总成交额", "净流入", "上涨家数", "下跌家数", "领涨股-涨跌幅"]:
        if col in out.columns:
            out[col] = numeric_series(out[col])
    out = out.rename(
        columns={
            "板块": "sector",
            "涨跌幅": "pct_chg",
            "总成交额": "amount",
            "净流入": "net_inflow",
            "上涨家数": "up_count",
            "下跌家数": "down_count",
            "领涨股": "leader",
            "领涨股-涨跌幅": "leader_pct",
        }
    )
    if "up_ratio" not in out.columns and {"up_count", "down_count"}.issubset(out.columns):
        out["up_ratio"] = out["up_count"] / (out["up_count"] + out["down_count"]).replace(0, np.nan) * 100
    if "rank_pct" not in out.columns and "pct_chg" in out.columns:
        out["rank_pct"] = out["pct_chg"].rank(ascending=False, method="min")
    if "rank_amount" not in out.columns and "amount" in out.columns:
        out["rank_amount"] = out["amount"].rank(ascending=False, method="min")
    if "rank_flow" not in out.columns and "net_inflow" in out.columns:
        out["rank_flow"] = out["net_inflow"].rank(ascending=False, method="min")
    return out, status


@st.cache_data(ttl=30, show_spinner=False)
def get_otc_watch_snapshot_from_db() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    return load_latest_otc_watch_snapshot()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "amount" not in out.columns or out["amount"].isna().all():
        if "volume" in out.columns:
            out["amount"] = out["volume"] * out["close"]
        else:
            out["amount"] = np.nan
    out["pct"] = out["close"].pct_change() * 100
    for window in [5, 10, 20, 60]:
        out[f"ma{window}"] = out["close"].rolling(window).mean()
        out[f"amount_ma{window}"] = out["amount"].rolling(window).mean()
    out["ema12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema12"] - out["ema26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (out["macd"] - out["macd_signal"]) * 2

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)

    typical = (out["high"] + out["low"] + out["close"]) / 3
    money_flow = typical * out["volume"]
    positive = money_flow.where(typical.diff() > 0, 0).rolling(14).sum()
    negative = money_flow.where(typical.diff() < 0, 0).rolling(14).sum().abs()
    out["mfi14"] = 100 - 100 / (1 + positive / negative.replace(0, np.nan))

    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift()).abs(),
            (out["low"] - out["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = true_range.rolling(14).mean()
    direction = np.sign(out["close"].diff()).fillna(0)
    out["obv"] = (direction * out["volume"]).cumsum()
    out["obv_ma5"] = out["obv"].rolling(5).mean()
    out["high20"] = out["high"].rolling(20).max()
    out["high60"] = out["high"].rolling(60).max()
    out["low20"] = out["low"].rolling(20).min()
    out["drawdown20"] = out["close"] / out["high"].rolling(20).max() - 1
    out["return5"] = out["close"].pct_change(5) * 100
    out["return10"] = out["close"].pct_change(10) * 100
    out["return20"] = out["close"].pct_change(20) * 100
    out["amount_ratio5"] = out["amount"] / out["amount_ma5"]
    out["amount_ratio20"] = out["amount"] / out["amount_ma20"]
    return out


def latest_spot_row(spot_df: pd.DataFrame | None, code: str) -> pd.Series | None:
    if spot_df is None or "代码" not in spot_df.columns:
        return None
    hit = spot_df[spot_df["代码"].astype(str).str.zfill(6) == code]
    return hit.iloc[0] if not hit.empty else None


def etf_label_from_row(row: pd.Series | dict[str, Any]) -> str:
    code = str(row.get("代码", "")).zfill(6)
    name = str(row.get("名称", "-"))
    pct = pct_text(to_num(row.get("涨跌幅")))
    amount = format_money(to_num(row.get("成交额")))
    return f"{code} | {name} | {pct} | {amount}"


def etf_label_for_code(spot_df: pd.DataFrame | None, code: str) -> str:
    row = latest_spot_row(spot_df, code)
    if row is not None:
        return etf_label_from_row(row)
    return f"{code} | 未取到实时名称"


def fund_label_from_row(row: pd.Series | dict[str, Any]) -> str:
    code = str(row.get("基金代码", "")).zfill(6)
    name = str(row.get("基金简称", "-"))
    fund_type = str(row.get("基金类型", ""))
    pct = pct_text(to_num(row.get("日增长率")))
    return " | ".join([part for part in [code, name, fund_type, pct] if part])


def latest_open_fund_row(daily_df: pd.DataFrame | None, code: str) -> pd.Series | None:
    if daily_df is None or daily_df.empty or "基金代码" not in daily_df.columns:
        return None
    hit = daily_df[daily_df["基金代码"].astype(str).str.zfill(6) == code]
    return hit.iloc[0] if not hit.empty else None


def open_fund_label_for_code(daily_df: pd.DataFrame | None, names_df: pd.DataFrame | None, code: str) -> str:
    row = latest_open_fund_row(daily_df, code)
    if row is not None:
        return fund_label_from_row(row)
    if names_df is not None and not names_df.empty and "基金代码" in names_df.columns:
        hit = names_df[names_df["基金代码"].astype(str).str.zfill(6) == code]
        if not hit.empty:
            return fund_label_from_row(hit.iloc[0])
    return f"{code} | 未取到基金名称"


def open_fund_name_for_code(
    daily_df: pd.DataFrame | None,
    names_df: pd.DataFrame | None,
    snapshot_row: pd.Series | None,
    estimation_row: pd.Series | None,
    code: str,
) -> str:
    if snapshot_row is not None:
        name = str(snapshot_row.get("基金简称") or snapshot_row.get("基金名称") or "").strip()
        if name and name != code:
            return name
    row = latest_open_fund_row(daily_df, code)
    if row is not None:
        name = str(row.get("基金简称") or row.get("基金名称") or "").strip()
        if name and name != code:
            return name
    if names_df is not None and not names_df.empty and "基金代码" in names_df.columns:
        names = names_df.copy()
        names["基金代码"] = names["基金代码"].astype(str).str.zfill(6)
        hit = names[names["基金代码"] == clean_code(code)]
        if not hit.empty:
            name = str(hit.iloc[0].get("基金简称") or hit.iloc[0].get("基金名称") or "").strip()
            if name and name != code:
                return name
    if estimation_row is not None:
        name = str(estimation_row.get("基金名称") or "").strip()
        if name and name != code:
            return name
    return clean_code(code)


def build_open_fund_search_options(
    daily_df: pd.DataFrame | None,
    names_df: pd.DataFrame | None,
    query: str,
    current_code: str,
    limit: int = 150,
) -> list[str]:
    current_code = clean_code(current_code)
    if daily_df is None or daily_df.empty:
        base = names_df.copy() if names_df is not None else pd.DataFrame()
    else:
        base = daily_df.copy()
        if names_df is not None and not names_df.empty:
            base["基金代码"] = base["基金代码"].astype(str).str.zfill(6)
            names = names_df.copy()
            names["基金代码"] = names["基金代码"].astype(str).str.zfill(6)
            base = base.merge(names[["基金代码", "基金类型", "拼音缩写", "拼音全称"]], on="基金代码", how="left")

    if base.empty or "基金代码" not in base.columns:
        return [f"{current_code} | 场外基金列表不可用"]
    base["基金代码"] = base["基金代码"].astype(str).str.zfill(6)
    for col in ["基金简称", "基金类型", "拼音缩写", "拼音全称"]:
        if col not in base.columns:
            base[col] = ""
    if "日增长率" in base.columns:
        base["日增长率"] = numeric_series(base["日增长率"])
    else:
        base["日增长率"] = np.nan

    query = (query or "").strip()
    if query:
        mask = (
            base["基金代码"].astype(str).str.contains(query, case=False, na=False)
            | base["基金简称"].astype(str).str.contains(query, case=False, na=False)
            | base["基金类型"].astype(str).str.contains(query, case=False, na=False)
            | base["拼音缩写"].astype(str).str.contains(query, case=False, na=False)
            | base["拼音全称"].astype(str).str.contains(query, case=False, na=False)
        )
        filtered = base[mask].copy()
        filtered["基金代码"] = filtered["基金代码"].astype(str).str.zfill(6)
        filtered["_rank"] = filtered["基金代码"].astype(str).str.startswith(str(query)).astype(int) * 3 + filtered["基金简称"].astype(str).str.contains(query, case=False, na=False).astype(int)
        filtered = filtered.sort_values(["_rank", "日增长率", "基金代码"], ascending=[False, False, True], na_position="last")
    else:
        filtered = base.sort_values(["日增长率", "基金代码"], ascending=[False, True], na_position="last")

    current = base[base["基金代码"] == current_code]
    if not current.empty and current_code not in set(filtered["基金代码"].head(limit).tolist()):
        filtered = pd.concat([current, filtered], ignore_index=True)
    filtered = filtered.drop_duplicates("基金代码").head(limit)
    return [fund_label_from_row(row) for _, row in filtered.iterrows()]


def build_etf_search_options(
    spot_df: pd.DataFrame | None,
    query: str,
    current_code: str,
    limit: int = 120,
) -> list[str]:
    current_code = clean_code(current_code)
    if spot_df is None or spot_df.empty or "代码" not in spot_df.columns:
        return [f"{current_code} | 实时ETF列表不可用"]

    df = spot_df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    if "名称" not in df.columns:
        df["名称"] = ""
    for col in ["成交额", "涨跌幅"]:
        if col in df.columns:
            df[col] = numeric_series(df[col])

    query = (query or "").strip()
    if query:
        mask = df["代码"].str.contains(query, case=False, na=False) | df["名称"].astype(str).str.contains(query, case=False, na=False)
        filtered = df[mask].copy()
        filtered["_rank"] = (
            filtered["代码"].str.startswith(query).astype(int) * 3
            + filtered["名称"].astype(str).str.contains(query, case=False, na=False).astype(int)
        )
        filtered = filtered.sort_values(["_rank", "成交额"], ascending=[False, False], na_position="last")
    else:
        filtered = df.sort_values("成交额", ascending=False, na_position="last")

    current = df[df["代码"] == current_code]
    if not current.empty and current_code not in set(filtered["代码"].head(limit).tolist()):
        filtered = pd.concat([current, filtered], ignore_index=True)
    filtered = filtered.drop_duplicates("代码").head(limit)
    return [etf_label_from_row(row) for _, row in filtered.iterrows()]


def infer_sector_name(
    etf_name: str,
    track_target: str,
    custom_keyword: str,
    sector_df: pd.DataFrame | None,
) -> str | None:
    if sector_df is None or sector_df.empty:
        return None
    sectors = sector_df["sector"].dropna().astype(str).tolist()
    if custom_keyword:
        custom = custom_keyword.strip()
        for sector in sectors:
            if custom in sector or sector in custom:
                return sector
    haystack = f"{etf_name} {track_target}"
    for pattern, mapped in THEME_TO_SECTOR:
        if re.search(pattern, haystack, flags=re.I):
            for sector in sectors:
                if mapped in sector or sector in mapped:
                    return sector
    return None


def overview_fields(overview: pd.DataFrame | None) -> dict[str, str]:
    if overview is None or overview.empty:
        return {}
    row = overview.iloc[0].to_dict()
    return {str(k): str(v) for k, v in row.items()}


def compute_market_env(index_df: pd.DataFrame | None, a_spot: pd.DataFrame | None, etf_spot: pd.DataFrame | None) -> dict[str, Any]:
    score = 50.0
    breadth_label = "无实时宽度"
    breadth = np.nan
    index_label = "指数数据不足"
    if index_df is not None and len(index_df) >= 65:
        idx = add_indicators(index_df)
        last = idx.iloc[-1]
        score = 0
        score += 20 if last["close"] > last["ma20"] else 5
        score += 20 if last["close"] > last["ma60"] else 5
        score += 18 if last["ma20"] > idx["ma20"].iloc[-6] else 6
        score += linear_score(last.get("return20", np.nan), -6, 8) * 0.22
        score += linear_score(last.get("return5", np.nan), -3, 4) * 0.20
        score = clamp(score)
        index_label = f"指数收盘 {last['close']:.2f}，20日收益 {last.get('return20', np.nan):.2f}%"

    if a_spot is not None and "涨跌幅" in a_spot.columns:
        pct = numeric_series(a_spot["涨跌幅"])
        up = (pct > 0).sum()
        down = (pct < 0).sum()
        breadth = up / max(up + down, 1) * 100
        breadth_label = f"A股上涨占比 {breadth:.1f}%"
        score = clamp(score * 0.72 + linear_score(breadth, 35, 68) * 0.28)
    elif etf_spot is not None and "涨跌幅" in etf_spot.columns:
        pct = numeric_series(etf_spot["涨跌幅"])
        up = (pct > 0).sum()
        down = (pct < 0).sum()
        breadth = up / max(up + down, 1) * 100
        breadth_label = f"ETF上涨占比 {breadth:.1f}%"
        score = clamp(score * 0.75 + linear_score(breadth, 35, 68) * 0.25)

    if score >= 65:
        regime = "顺风"
    elif score >= 45:
        regime = "震荡"
    else:
        regime = "逆风"
    return {"score": score, "regime": regime, "index_label": index_label, "breadth_label": breadth_label, "breadth": breadth}


def sector_heat_score(sector_df: pd.DataFrame | None, sector_name: str | None) -> tuple[float, dict[str, Any]]:
    if sector_df is None or sector_df.empty:
        return 50.0, {"sector": "无板块数据", "note": "行业接口不可用"}
    n = max(len(sector_df), 1)
    if sector_name:
        hit = sector_df[sector_df["sector"].astype(str) == sector_name]
    else:
        hit = pd.DataFrame()
    if hit.empty:
        top = sector_df.sort_values(["pct_chg", "net_inflow"], ascending=[False, False]).iloc[0]
        return 52.0, {
            "sector": "未匹配具体行业",
            "proxy_sector": top["sector"],
            "note": f"当前最热行业为 {top['sector']}，ETF按全市场热度中性处理",
        }
    row = hit.iloc[0]
    pct_rank = (n - to_num(row["rank_pct"]) + 1) / n * 100
    amount_rank = (n - to_num(row["rank_amount"]) + 1) / n * 100
    flow_rank = (n - to_num(row["rank_flow"]) + 1) / n * 100
    up_ratio_score = linear_score(row.get("up_ratio", np.nan), 35, 82)
    pct_strength = linear_score(row.get("pct_chg", np.nan), -2.5, 4.5)
    score = clamp(pct_rank * 0.30 + amount_rank * 0.22 + flow_rank * 0.24 + up_ratio_score * 0.16 + pct_strength * 0.08)
    return score, row.to_dict()


def etf_heat_proxy(spot_row: pd.Series | None, spot_df: pd.DataFrame | None) -> float:
    if spot_row is None or spot_df is None or spot_df.empty:
        return 50.0
    score = 0
    score += percentile_score(to_num(spot_row.get("涨跌幅")), spot_df["涨跌幅"]) * 0.25
    score += percentile_score(to_num(spot_row.get("成交额")), spot_df["成交额"]) * 0.25
    if "主力净流入-净占比" in spot_df.columns:
        score += percentile_score(to_num(spot_row.get("主力净流入-净占比")), spot_df["主力净流入-净占比"]) * 0.25
    else:
        score += 50 * 0.25
    score += soft_ratio_score(to_num(spot_row.get("量比")), 1.05, 2.5, 0.4, 4.0) * 0.15
    premium = abs(to_num(spot_row.get("基金折价率")))
    score += linear_score(premium, 2.0, 0.0) * 0.10
    return clamp(score)


def score_model(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None,
    spot_row: pd.Series | None,
    etf_spot: pd.DataFrame | None,
    sector_df: pd.DataFrame | None,
    sector_name: str | None,
    market_env: dict[str, Any],
) -> dict[str, Any]:
    ind = add_indicators(df)
    idx = add_indicators(index_df) if index_df is not None and len(index_df) >= 65 else None
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) > 1 else last
    close = last["close"]

    above_ma = sum(close > last.get(f"ma{w}", np.nan) for w in [5, 10, 20, 60])
    alignment = sum(
        [
            last.get("ma5", np.nan) > last.get("ma10", np.nan),
            last.get("ma10", np.nan) > last.get("ma20", np.nan),
            last.get("ma20", np.nan) > last.get("ma60", np.nan),
        ]
    )
    slope5 = (last["ma5"] / ind["ma5"].iloc[-6] - 1) * 100 if len(ind) >= 6 and pd.notna(ind["ma5"].iloc[-6]) else np.nan
    slope20 = (last["ma20"] / ind["ma20"].iloc[-6] - 1) * 100 if len(ind) >= 6 and pd.notna(ind["ma20"].iloc[-6]) else np.nan
    breakout20 = close / last.get("high20", np.nan) - 1
    breakout60 = close / last.get("high60", np.nan) - 1
    rel_strength = np.nan
    if idx is not None and len(idx) >= 21:
        etf_ret20 = last.get("return20", np.nan)
        idx_ret20 = idx.iloc[-1].get("return20", np.nan)
        rel_strength = etf_ret20 - idx_ret20
    trend_score = clamp(
        above_ma / 4 * 28
        + alignment / 3 * 22
        + linear_score(slope5, -1.5, 3.5) * 0.14
        + linear_score(slope20, -2.0, 4.0) * 0.12
        + linear_score(breakout20 * 100, -8, 2.5) * 0.12
        + linear_score(breakout60 * 100, -12, 1.5) * 0.06
        + linear_score(rel_strength, -6, 8) * 0.06
    )

    amount_ratio5 = to_num(last.get("amount_ratio5"))
    amount_ratio20 = to_num(last.get("amount_ratio20"))
    lb = to_num(spot_row.get("量比")) if spot_row is not None else np.nan
    price_volume_confirm = 80 if last.get("pct", 0) >= 0 and amount_ratio5 >= 1.05 else 42
    if last.get("pct", 0) < 0 and amount_ratio5 >= 1.4:
        price_volume_confirm = 28
    volume_score = clamp(
        soft_ratio_score(amount_ratio5, 1.05, 2.2, 0.55, 4.0) * 0.34
        + soft_ratio_score(amount_ratio20, 1.0, 2.0, 0.5, 4.2) * 0.28
        + soft_ratio_score(lb, 1.0, 2.6, 0.4, 5.0) * 0.18
        + price_volume_confirm * 0.20
    )

    main_ratio = to_num(spot_row.get("主力净流入-净占比")) if spot_row is not None else np.nan
    main_amount = to_num(spot_row.get("主力净流入-净额")) if spot_row is not None else np.nan
    amount_today = to_num(spot_row.get("成交额")) if spot_row is not None else to_num(last.get("amount"))
    if pd.isna(main_ratio) and not pd.isna(main_amount) and amount_today:
        main_ratio = main_amount / amount_today * 100
    main_rank = percentile_score(main_amount, etf_spot["主力净流入-净额"]) if etf_spot is not None and "主力净流入-净额" in etf_spot.columns else 50
    obv_slope = (last["obv"] / ind["obv_ma5"].iloc[-1] - 1) * 100 if pd.notna(ind["obv_ma5"].iloc[-1]) and ind["obv_ma5"].iloc[-1] else np.nan
    accumulation_days = int(((ind["pct"].tail(5) > 0) & (ind["amount_ratio5"].tail(5) > 0.95)).sum())
    funds_score = clamp(
        linear_score(main_ratio, -7, 9) * 0.38
        + main_rank * 0.20
        + linear_score(last.get("mfi14", np.nan), 32, 75) * 0.18
        + linear_score(obv_slope, -8, 12) * 0.14
        + linear_score(accumulation_days, 0, 4) * 0.10
    )

    sector_score, sector_info = sector_heat_score(sector_df, sector_name)
    proxy_score = etf_heat_proxy(spot_row, etf_spot)
    if sector_info.get("sector") == "未匹配具体行业":
        sector_score = clamp(proxy_score * 0.65 + sector_score * 0.35)

    high60 = last.get("high60", np.nan)
    space_to_high = (high60 / close - 1) * 100 if pd.notna(high60) and close else np.nan
    premium = to_num(spot_row.get("基金折价率")) if spot_row is not None else np.nan
    atr_pct = last.get("atr14", np.nan) / close * 100 if close else np.nan
    ret5 = last.get("return5", np.nan)
    ret10 = last.get("return10", np.nan)
    drawdown20 = abs(last.get("drawdown20", np.nan) * 100)
    overheat_penalty_score = 100
    if ret5 > 8:
        overheat_penalty_score -= min(32, (ret5 - 8) * 4)
    if ret10 > 15:
        overheat_penalty_score -= min(26, (ret10 - 15) * 2.5)
    if amount_ratio5 > 2.6:
        overheat_penalty_score -= min(24, (amount_ratio5 - 2.6) * 12)
    if abs(premium) > 0.6:
        overheat_penalty_score -= min(18, (abs(premium) - 0.6) * 18)

    risk_score = clamp(
        soft_ratio_score(space_to_high, 2, 15, -6, 32) * 0.22
        + linear_score(drawdown20, 15, 2, reverse=False) * 0.18
        + linear_score(abs(premium), 1.8, 0, reverse=False) * 0.22
        + clamp(overheat_penalty_score) * 0.26
        + linear_score(atr_pct, 8, 1.5, reverse=False) * 0.12
    )

    factor_scores = {
        "趋势分": trend_score,
        "量能分": volume_score,
        "资金分": funds_score,
        "板块热度分": sector_score,
        "风险控制分": risk_score,
    }
    total = (
        trend_score * WEIGHTS["trend"]
        + volume_score * WEIGHTS["volume"]
        + funds_score * WEIGHTS["funds"]
        + sector_score * WEIGHTS["sector"]
        + risk_score * WEIGHTS["risk"]
    )

    high_volume_stall = amount_ratio5 > 1.7 and last.get("pct", 0) < 0.8 and close < (last["high"] + last["low"]) / 2
    overheat = ret5 > 10 and amount_ratio5 > 2.0 and abs(premium) > 0.45
    broken_ma10 = close < last.get("ma10", np.nan)
    broken_ma20 = close < last.get("ma20", np.nan)
    main_outflow = main_ratio < -2 or (not pd.isna(main_amount) and main_amount < 0)

    if overheat and risk_score < 55:
        action = "禁止追高"
        action_tone = "risk"
    elif broken_ma20 and (sector_score < 48 or funds_score < 45):
        action = "离场"
        action_tone = "risk"
    elif high_volume_stall or (broken_ma10 and main_outflow):
        action = "减仓"
        action_tone = "warn"
    elif total >= 72 and trend_score >= 62 and volume_score >= 55 and funds_score >= 52 and risk_score >= 48 and market_env["score"] >= 42:
        action = "买入观察"
        action_tone = "good"
    elif total >= 58 and not broken_ma20 and not main_outflow:
        action = "持有"
        action_tone = "neutral"
    else:
        action = "观察等待"
        action_tone = "neutral"

    if market_env["score"] < 38 and action in {"买入观察", "持有"}:
        action = f"{action}（轻仓）"
        action_tone = "warn"

    positives: list[str] = []
    negatives: list[str] = []
    if above_ma >= 3 and alignment >= 2:
        positives.append("趋势结构较完整，短中期均线形成支撑")
    elif broken_ma20:
        negatives.append("价格跌破20日线，短线趋势证据不足")
    if amount_ratio5 >= 1.1 and last.get("pct", 0) >= 0:
        positives.append(f"成交额为5日均值的 {amount_ratio5:.2f} 倍，量价确认较好")
    elif amount_ratio5 < 0.8:
        negatives.append("成交额低于5日均值，启动信号偏弱")
    if main_ratio > 2:
        positives.append(f"主力净流入占成交额约 {main_ratio:.2f}%")
    elif main_ratio < -2:
        negatives.append(f"主力净流出占成交额约 {abs(main_ratio):.2f}%")
    if sector_score >= 65:
        positives.append("匹配板块处于涨幅/资金/成交额前排")
    elif sector_score < 45:
        negatives.append("板块热度或ETF相对热度偏弱")
    if overheat:
        negatives.append("短期涨幅、放量与溢价同时升高，追高性价比下降")
    if risk_score >= 65:
        positives.append("前高空间、波动和溢价约束仍可接受")
    elif risk_score < 45:
        negatives.append("风险控制分偏低，需缩小仓位或等待回撤确认")

    details = {
        "latest": last.to_dict(),
        "prev": prev.to_dict(),
        "factor_scores": factor_scores,
        "total_score": clamp(total),
        "action": action,
        "action_tone": action_tone,
        "sector_info": sector_info,
        "sector_name": sector_name,
        "market_env": market_env,
        "positives": positives,
        "negatives": negatives,
        "raw": {
            "amount_ratio5": amount_ratio5,
            "amount_ratio20": amount_ratio20,
            "main_ratio": main_ratio,
            "main_amount": main_amount,
            "premium": premium,
            "space_to_high": space_to_high,
            "ret5": ret5,
            "ret10": ret10,
            "drawdown20": drawdown20,
            "atr_pct": atr_pct,
            "relative_strength_20d": rel_strength,
            "accumulation_days": accumulation_days,
        },
    }
    return details


def otc_nav_to_price_df(nav_df: pd.DataFrame | None) -> pd.DataFrame:
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

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": numeric_series(df[nav_col]),
        }
    ).dropna(subset=["date", "close"])
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if "日增长率" in df.columns:
        pct_map = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "vendor_pct": numeric_series(df["日增长率"])})
        out = out.merge(pct_map.dropna(subset=["date"]).drop_duplicates("date"), on="date", how="left")
    else:
        out["vendor_pct"] = np.nan
    out["pct"] = out["vendor_pct"].combine_first(out["close"].pct_change() * 100)
    prev_close = out["close"].shift().fillna(out["close"])
    out["open"] = prev_close
    out["high"] = pd.concat([prev_close, out["close"]], axis=1).max(axis=1)
    out["low"] = pd.concat([prev_close, out["close"]], axis=1).min(axis=1)
    out["volume"] = 1.0
    out["amount"] = (out["pct"].abs().fillna(0) + 1) * 10000
    return out[["date", "open", "high", "low", "close", "volume", "amount", "pct"]]


def achievement_value(df: pd.DataFrame | None, period: str, col: str) -> float:
    if df is None or df.empty or "周期" not in df.columns or col not in df.columns:
        return np.nan
    hit = df[df["周期"].astype(str) == period]
    if hit.empty:
        return np.nan
    return to_num(hit.iloc[0].get(col))


def achievement_rank_score(df: pd.DataFrame | None, period: str) -> float:
    if df is None or df.empty or "周期" not in df.columns or "周期收益同类排名" not in df.columns:
        return 50.0
    hit = df[df["周期"].astype(str) == period]
    if hit.empty:
        return 50.0
    text = str(hit.iloc[0].get("周期收益同类排名", ""))
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if not match:
        return 50.0
    rank = float(match.group(1))
    total = max(float(match.group(2)), 1.0)
    return clamp((total - rank + 1) / total * 100)


def asset_position(asset_df: pd.DataFrame | None, keyword: str) -> float:
    if asset_df is None or asset_df.empty or "资产类型" not in asset_df.columns or "仓位占比" not in asset_df.columns:
        return np.nan
    hit = asset_df[asset_df["资产类型"].astype(str).str.contains(keyword, na=False)]
    if hit.empty:
        return np.nan
    return to_num(hit.iloc[0].get("仓位占比"))


def score_open_fund_model(
    nav_df: pd.DataFrame | None,
    index_df: pd.DataFrame | None,
    open_fund_daily: pd.DataFrame | None,
    estimation_df: pd.DataFrame | None,
    otc_row: pd.Series | None,
    estimation_row: pd.Series | None,
    achievement_df: pd.DataFrame | None,
    asset_df: pd.DataFrame | None,
    holding_impact: pd.DataFrame | None,
    sector_df: pd.DataFrame | None,
    sector_name: str | None,
    market_env: dict[str, Any],
) -> dict[str, Any]:
    price_df = otc_nav_to_price_df(nav_df)
    through_nav = to_num(row.get("实时穿透净值"))
    if pd.isna(through_nav):
        through_nav = to_num(row.get("估算净值"))
    through_pct = to_num(row.get("实时穿透涨幅"))
    if pd.isna(through_pct):
        through_pct = to_num(row.get("估算涨幅"))
    if pd.notna(through_nav):
        snapshot_dt = pd.to_datetime(row.get("快照时间"), errors="coerce")
        snapshot_dt = snapshot_dt if pd.notna(snapshot_dt) else pd.Timestamp(date.today())
        append_row = pd.DataFrame(
            {
                "date": [snapshot_dt.normalize()],
                "open": [through_nav],
                "high": [through_nav],
                "low": [through_nav],
                "close": [through_nav],
                "volume": [1.0],
                "amount": [(abs(through_pct) if pd.notna(through_pct) else 1.0) * 10000],
                "pct": [through_pct],
            }
        )
        if price_df.empty:
            price_df = append_row
        else:
            price_df = pd.concat([price_df, append_row], ignore_index=True)
            price_df = price_df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    ind = add_indicators(price_df) if len(price_df) >= 2 else pd.DataFrame()
    idx = add_indicators(index_df) if index_df is not None and len(index_df) >= 65 else None
    has_nav = not ind.empty
    last = ind.iloc[-1] if has_nav else pd.Series(dtype="float64")
    close = to_num(last.get("close"))

    above_ma = sum(close > to_num(last.get(f"ma{w}")) for w in [5, 10, 20, 60]) if has_nav else 0
    alignment = sum(
        [
            to_num(last.get("ma5")) > to_num(last.get("ma10")),
            to_num(last.get("ma10")) > to_num(last.get("ma20")),
            to_num(last.get("ma20")) > to_num(last.get("ma60")),
        ]
    ) if has_nav else 0
    slope5 = (last["ma5"] / ind["ma5"].iloc[-6] - 1) * 100 if len(ind) >= 6 and pd.notna(ind["ma5"].iloc[-6]) and ind["ma5"].iloc[-6] else np.nan
    slope20 = (last["ma20"] / ind["ma20"].iloc[-6] - 1) * 100 if len(ind) >= 26 and pd.notna(ind["ma20"].iloc[-6]) and ind["ma20"].iloc[-6] else np.nan
    breakout20 = (close / to_num(last.get("high20")) - 1) * 100 if has_nav and to_num(last.get("high20")) else np.nan
    breakout60 = (close / to_num(last.get("high60")) - 1) * 100 if has_nav and to_num(last.get("high60")) else np.nan
    rel_strength = np.nan
    if idx is not None and len(idx) >= 21 and has_nav:
        rel_strength = to_num(last.get("return20")) - to_num(idx.iloc[-1].get("return20"))
    trend_score = clamp(
        above_ma / 4 * 28
        + alignment / 3 * 22
        + linear_score(slope5, -1.5, 3.0) * 0.14
        + linear_score(slope20, -2.5, 4.5) * 0.12
        + linear_score(breakout20, -8, 1.5) * 0.12
        + linear_score(breakout60, -15, 1.0) * 0.06
        + linear_score(rel_strength, -7, 8) * 0.06
    )

    estimate_pct = to_num(estimation_row.get("估算涨幅")) if estimation_row is not None else np.nan
    daily_pct = estimate_pct if pd.notna(estimate_pct) else (to_num(otc_row.get("日增长率")) if otc_row is not None else to_num(last.get("pct")))
    ret5 = to_num(last.get("return5"))
    ret10 = to_num(last.get("return10"))
    ret20 = to_num(last.get("return20"))
    positive_days5 = int((ind["pct"].tail(5) > 0).sum()) if has_nav and "pct" in ind.columns else 0
    month_ret = achievement_value(achievement_df, "近1月", "本产品区间收益")
    quarter_ret = achievement_value(achievement_df, "近3月", "本产品区间收益")
    ytd_ret = achievement_value(achievement_df, "今年以来", "本产品区间收益")
    achievement_rank = np.mean([achievement_rank_score(achievement_df, p) for p in ["近1月", "近3月", "今年以来"]])
    momentum_score = clamp(
        linear_score(daily_pct, -2.5, 2.8) * 0.16
        + linear_score(ret5, -4.5, 6.5) * 0.18
        + linear_score(ret20, -9, 13) * 0.16
        + linear_score(positive_days5, 1, 4) * 0.12
        + linear_score(month_ret, -10, 10) * 0.13
        + linear_score(quarter_ret, -16, 20) * 0.12
        + linear_score(ytd_ret, -28, 35) * 0.07
        + achievement_rank * 0.06
    )

    stock_position = asset_position(asset_df, "股票")
    top_weight = np.nan
    contribution = np.nan
    up_ratio = np.nan
    flow_score = 50.0
    if holding_impact is not None and not holding_impact.empty:
        if "占净值比例" in holding_impact.columns:
            top_weight = numeric_series(holding_impact["占净值比例"]).dropna().sum()
        if "估算贡献" in holding_impact.columns:
            contribution = numeric_series(holding_impact["估算贡献"]).dropna().sum()
        if "涨跌幅" in holding_impact.columns:
            pct = numeric_series(holding_impact["涨跌幅"]).dropna()
            if not pct.empty:
                up_ratio = (pct > 0).mean() * 100
        if "主力净流入-净额" in holding_impact.columns:
            flow = numeric_series(holding_impact["主力净流入-净额"]).dropna()
            if not flow.empty:
                flow_score = linear_score(flow.sum(), -8e8, 8e8)
    holding_score = clamp(
        linear_score(contribution, -1.1, 1.2) * 0.38
        + linear_score(up_ratio, 32, 70) * 0.22
        + flow_score * 0.16
        + linear_score(stock_position, 35, 92) * 0.12
        + soft_ratio_score(top_weight, 28, 68, 5, 90) * 0.12
    )

    sector_score, sector_info = sector_heat_score(sector_df, sector_name)
    if estimation_df is not None and "估算涨幅" in estimation_df.columns and estimation_df["估算涨幅"].notna().any() and pd.notna(estimate_pct):
        peer_rank = percentile_score(estimate_pct, estimation_df["估算涨幅"])
    else:
        peer_rank = percentile_score(daily_pct, open_fund_daily["日增长率"]) if open_fund_daily is not None and "日增长率" in open_fund_daily.columns else 50.0
    peer_score = clamp(peer_rank * 0.36 + sector_score * 0.28 + achievement_rank * 0.22 + market_env.get("score", 50) * 0.14)

    high60 = to_num(last.get("high60"))
    space_to_high = (high60 / close - 1) * 100 if high60 and close else np.nan
    nav_high120 = ind["close"].rolling(120, min_periods=20).max().iloc[-1] if has_nav and len(ind) >= 20 else np.nan
    drawdown120 = abs((close / nav_high120 - 1) * 100) if pd.notna(nav_high120) and nav_high120 and close else np.nan
    volatility20 = ind["pct"].tail(20).std() if has_nav and "pct" in ind.columns else np.nan
    max_dd_1y = achievement_value(achievement_df, "近1年", "本产品最大回撒")
    max_dd_3m = achievement_value(achievement_df, "近3月", "本产品最大回撒")
    max_dd = max([v for v in [max_dd_1y, max_dd_3m] if pd.notna(v)], default=np.nan)
    concentration_risk_score = linear_score(top_weight, 85, 35)
    stock_position_risk_score = 100 - max(0, stock_position - 92) * 2 if pd.notna(stock_position) else 50
    overheat_penalty_score = 100.0
    if ret5 > 7:
        overheat_penalty_score -= min(28, (ret5 - 7) * 4)
    if ret10 > 12:
        overheat_penalty_score -= min(24, (ret10 - 12) * 2.5)
    if daily_pct > 3.5:
        overheat_penalty_score -= min(18, (daily_pct - 3.5) * 5)
    risk_score = clamp(
        soft_ratio_score(space_to_high, 2, 16, -8, 35) * 0.18
        + linear_score(drawdown120, 22, 3) * 0.20
        + linear_score(volatility20, 4.2, 0.6) * 0.15
        + linear_score(max_dd, 35, 8) * 0.20
        + clamp(overheat_penalty_score) * 0.15
        + concentration_risk_score * 0.07
        + clamp(stock_position_risk_score) * 0.05
    )

    factor_scores = {
        "趋势分": trend_score,
        "净值动能分": momentum_score,
        "持仓穿透分": holding_score,
        "同类热度分": peer_score,
        "风险控制分": risk_score,
    }
    total = (
        trend_score * WEIGHTS["trend"]
        + momentum_score * WEIGHTS["volume"]
        + holding_score * WEIGHTS["funds"]
        + peer_score * WEIGHTS["sector"]
        + risk_score * WEIGHTS["risk"]
    )

    broken_ma20 = has_nav and close < to_num(last.get("ma20"))
    broken_ma60 = has_nav and close < to_num(last.get("ma60"))
    holding_drag = pd.notna(contribution) and contribution < -0.30 and (pd.isna(up_ratio) or up_ratio < 45)
    overheat = ret5 > 8 and ret10 > 12 and risk_score < 55

    if overheat:
        action = "禁止追高"
        action_tone = "risk"
    elif broken_ma60 and (peer_score < 48 or holding_score < 45):
        action = "离场"
        action_tone = "risk"
    elif broken_ma20 or holding_drag or drawdown120 > 12:
        action = "减仓"
        action_tone = "warn"
    elif total >= 72 and trend_score >= 62 and momentum_score >= 55 and holding_score >= 50 and risk_score >= 48 and market_env.get("score", 50) >= 42:
        action = "买入观察"
        action_tone = "good"
    elif total >= 58 and not broken_ma20 and not holding_drag:
        action = "持有"
        action_tone = "neutral"
    else:
        action = "观察等待"
        action_tone = "neutral"

    if market_env.get("score", 50) < 38 and action in {"买入观察", "持有"}:
        action = f"{action}（轻仓）"
        action_tone = "warn"

    positives: list[str] = []
    negatives: list[str] = []
    if above_ma >= 3 and alignment >= 2:
        positives.append("净值站上多条均线，趋势结构较完整")
    elif broken_ma20:
        negatives.append("净值跌破20日均线，短线趋势需要修复")
    if momentum_score >= 65:
        positives.append("近端净值动能和阶段业绩处于较强区间")
    elif momentum_score < 45:
        negatives.append("近端净值动能偏弱，阶段收益或同类排名缺少确认")
    if pd.notna(contribution) and contribution > 0.15:
        positives.append(f"前20重仓股实时估算贡献约 {contribution:.2f}%")
    elif holding_drag:
        negatives.append("重仓股实时拖累较明显，持仓穿透未确认")
    if peer_rank >= 70 or sector_score >= 65:
        positives.append("同类净值表现或映射板块热度处于前排")
    elif peer_score < 45:
        negatives.append("同类排名、板块热度或市场环境偏弱")
    if risk_score >= 65:
        positives.append("回撤、波动、集中度和短期过热约束仍可接受")
    elif risk_score < 45:
        negatives.append("位置或回撤风险偏高，宜等待净值回撤/修复确认")

    return {
        "price_df": price_df,
        "factor_scores": factor_scores,
        "total_score": clamp(total),
        "action": action,
        "action_tone": action_tone,
        "sector_info": sector_info,
        "sector_name": sector_name,
        "market_env": market_env,
        "positives": positives,
        "negatives": negatives,
        "raw": {
            "daily_pct": daily_pct,
            "estimate_pct": estimate_pct,
            "estimated_nav": to_num(estimation_row.get("估算净值")) if estimation_row is not None else np.nan,
            "estimate_bias": to_num(estimation_row.get("估算偏差")) if estimation_row is not None else np.nan,
            "ret5": ret5,
            "ret10": ret10,
            "ret20": ret20,
            "month_ret": month_ret,
            "quarter_ret": quarter_ret,
            "ytd_ret": ytd_ret,
            "peer_rank": peer_rank,
            "achievement_rank": achievement_rank,
            "stock_position": stock_position,
            "top_weight": top_weight,
            "holding_contribution": contribution,
            "holding_up_ratio": up_ratio,
            "space_to_high": space_to_high,
            "drawdown120": drawdown120,
            "volatility20": volatility20,
            "max_drawdown": max_dd,
        },
    }


def build_etf_leaderboard(spot_df: pd.DataFrame | None, limit: int = 30) -> pd.DataFrame:
    if spot_df is None or spot_df.empty:
        return pd.DataFrame()
    df = spot_df.copy()
    for col in ["涨跌幅", "成交额", "主力净流入-净额", "主力净流入-净占比", "量比", "基金折价率", "换手率"]:
        if col in df.columns:
            df[col] = numeric_series(df[col])
    rank_cols = {
        "涨跌幅": 0.24,
        "成交额": 0.22,
        "主力净流入-净占比": 0.24,
        "主力净流入-净额": 0.16,
        "量比": 0.09,
    }
    score = pd.Series(0.0, index=df.index)
    for col, weight in rank_cols.items():
        if col in df.columns:
            score += df[col].rank(pct=True) * 100 * weight
        else:
            score += 50 * weight
    if "基金折价率" in df.columns:
        premium_score = (1 - (df["基金折价率"].abs() / 2.0).clip(0, 1)) * 100
        score += premium_score * 0.05
    df["实时机会分"] = score.clip(0, 100)
    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额", "量比", "基金折价率", "主力净流入-净额", "主力净流入-净占比", "实时机会分", "更新时间"]
    cols = [col for col in cols if col in df.columns]
    return df.sort_values("实时机会分", ascending=False)[cols].head(limit)


def build_realtime_etf_table(
    spot_df: pd.DataFrame | None,
    query: str = "",
    codes: list[str] | None = None,
    limit: int = 300,
) -> pd.DataFrame:
    if spot_df is None or spot_df.empty or "代码" not in spot_df.columns:
        if codes:
            return pd.DataFrame({"代码": codes, "名称": ["未取到实时行情"] * len(codes)})
        return pd.DataFrame()

    df = spot_df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    if "名称" not in df.columns:
        df["名称"] = ""
    numeric_cols = ["最新价", "涨跌幅", "成交额", "量比", "基金折价率", "换手率", "主力净流入-净额", "主力净流入-净占比"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = numeric_series(df[col])

    score = pd.Series(0.0, index=df.index)
    score_weights = {
        "涨跌幅": 0.24,
        "成交额": 0.22,
        "主力净流入-净占比": 0.24,
        "主力净流入-净额": 0.16,
        "量比": 0.09,
    }
    for col, weight in score_weights.items():
        if col in df.columns:
            score += df[col].rank(pct=True) * 100 * weight
        else:
            score += 50 * weight
    if "基金折价率" in df.columns:
        score += (1 - (df["基金折价率"].abs() / 2.0).clip(0, 1)) * 100 * 0.05
    df["实时机会分"] = score.clip(0, 100)

    pct = df.get("涨跌幅", pd.Series(np.nan, index=df.index))
    volume_ratio = df.get("量比", pd.Series(np.nan, index=df.index))
    main_ratio = df.get("主力净流入-净占比", pd.Series(np.nan, index=df.index))
    premium = df.get("基金折价率", pd.Series(np.nan, index=df.index))
    df["盯盘提示"] = np.select(
        [
            (pct > 5) & (volume_ratio > 2.2) & (premium.abs() > 0.5),
            (main_ratio > 3) & (volume_ratio > 1.1) & (pct > 0),
            (pct < -2) & (main_ratio < 0),
            (df["实时机会分"] >= 75),
        ],
        ["禁止追高", "资金关注", "弱势回避", "强势观察"],
        default="观察",
    )

    if codes is not None:
        codes = normalize_code_list(codes)
        order = {code: i for i, code in enumerate(codes)}
        df = df[df["代码"].isin(codes)].copy()
        missing = [code for code in codes if code not in set(df["代码"].tolist())]
        if missing:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame({"代码": missing, "名称": ["未取到实时行情"] * len(missing), "盯盘提示": ["待刷新"] * len(missing)}),
                ],
                ignore_index=True,
            )
        df["_order"] = df["代码"].map(order).fillna(9999)
        df = df.sort_values("_order")
    else:
        query = (query or "").strip()
        if query:
            mask = df["代码"].str.contains(query, case=False, na=False) | df["名称"].astype(str).str.contains(query, case=False, na=False)
            df = df[mask].copy()
        df = df.sort_values(["实时机会分", "成交额"], ascending=[False, False], na_position="last").head(limit)

    cols = ["代码", "名称", "最新价", "涨跌幅", "成交额", "量比", "基金折价率", "主力净流入-净额", "主力净流入-净占比", "实时机会分", "盯盘提示", "更新时间"]
    return df[[col for col in cols if col in df.columns]].reset_index(drop=True)


def format_realtime_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["成交额", "主力净流入-净额"]:
        if col in out:
            out[col] = out[col].map(format_money)
    for col in ["涨跌幅", "基金折价率", "主力净流入-净占比"]:
        if col in out:
            out[col] = out[col].map(lambda x: pct_text(x))
    for col in ["最新价", "量比", "实时机会分"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    return out


def build_open_fund_watch_table(
    daily_df: pd.DataFrame | None,
    names_df: pd.DataFrame | None,
    codes: list[str],
    estimation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    codes = normalize_code_list(codes)
    if not codes:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for code in codes:
        row = latest_open_fund_row(daily_df, code)
        payload = {"基金代码": code}
        if row is not None:
            payload.update(row.to_dict())
        elif names_df is not None and not names_df.empty and "基金代码" in names_df.columns:
            hit = names_df[names_df["基金代码"].astype(str).str.zfill(6) == code]
            if not hit.empty:
                payload.update(hit.iloc[0].to_dict())
        rows.append(payload)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["基金代码"] = out["基金代码"].astype(str).str.zfill(6)
        if names_df is not None and not names_df.empty and "基金类型" not in out.columns:
            names = names_df.copy()
            names["基金代码"] = names["基金代码"].astype(str).str.zfill(6)
            out = out.merge(names[["基金代码", "基金类型"]], on="基金代码", how="left")
        if estimation_df is not None and not estimation_df.empty and "基金代码" in estimation_df.columns:
            est = estimation_df.copy()
            est["基金代码"] = est["基金代码"].astype(str).str.zfill(6)
            est_cols = ["基金代码", "估算净值", "估算涨幅", "估算偏差", "公布单位净值", "公布日增长率", "上一净值", "估算日期"]
            est_cols = [col for col in est_cols if col in est.columns]
            out = out.merge(est[est_cols], on="基金代码", how="left")
        for col in out.columns:
            if "单位净值" in col or "累计净值" in col or col in {"日增长值", "日增长率", "估算净值", "估算涨幅", "估算偏差", "公布单位净值", "公布日增长率", "上一净值"}:
                out[col] = numeric_series(out[col])
        realtime_pct = out["估算涨幅"].combine_first(out["日增长率"]) if {"估算涨幅", "日增长率"}.issubset(out.columns) else out.get("日增长率", pd.Series(np.nan, index=out.index))
        if realtime_pct is not None:
            if estimation_df is not None and not estimation_df.empty and "估算涨幅" in estimation_df.columns and estimation_df["估算涨幅"].notna().any():
                benchmark = estimation_df["估算涨幅"]
            elif daily_df is not None and "日增长率" in daily_df.columns:
                benchmark = daily_df["日增长率"]
            else:
                benchmark = realtime_pct
            out["场外机会分"] = realtime_pct.map(lambda x: percentile_score(x, benchmark)).clip(0, 100)
            out["盯盘提示"] = np.select(
                [
                    realtime_pct >= 3.5,
                    realtime_pct <= -2.5,
                    out["场外机会分"] >= 75,
                    out["场外机会分"] <= 30,
                ],
                ["涨幅过快", "弱势回避", "强势观察", "同类偏弱"],
                default="观察",
            )
    return out


def format_open_fund_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if "单位净值" in col or "累计净值" in col:
            out[col] = out[col].map(lambda x: f"{to_num(x):.4f}" if pd.notna(to_num(x)) else "-")
    for col in ["日增长值", "日增长率"]:
        if col in out:
            out[col] = out[col].map(lambda x: pct_text(to_num(x)) if col == "日增长率" else (f"{to_num(x):.4f}" if pd.notna(to_num(x)) else "-"))
    for col in ["估算涨幅", "估算偏差", "公布日增长率", "实时穿透涨幅", "重仓估算贡献", "前20持仓权重", "上涨重仓股比例", "近5日", "近10日", "近20日", "120日回撤"]:
        if col in out:
            out[col] = out[col].map(lambda x: pct_text(to_num(x)))
    for col in ["估算净值", "实时穿透净值", "公布单位净值", "上一净值"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{to_num(x):.4f}" if pd.notna(to_num(x)) else "-")
    if "场外机会分" in out:
        out["场外机会分"] = out["场外机会分"].map(lambda x: f"{to_num(x):.1f}" if pd.notna(to_num(x)) else "-")
    for col in ["场外短线评分", "趋势分", "净值动能分", "持仓穿透分", "同类热度分", "风险控制分"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{to_num(x):.1f}" if pd.notna(to_num(x)) else "-")
    for col in ["DIF", "DEA", "MACD柱"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{to_num(x):.4f}" if pd.notna(to_num(x)) else "-")
    if "上涨重仓股数" in out:
        out["上涨重仓股数"] = out["上涨重仓股数"].map(lambda x: f"{to_num(x):.0f}" if pd.notna(to_num(x)) else "-")
    cols = ["基金代码", "基金简称", "基金类型", "场外短线评分", "动作", "估算涨幅", "实时穿透涨幅", "重仓估算贡献", "近5日", "近20日", "120日回撤"]
    cols += ["实时穿透净值", "估算净值", "估算偏差", "最新单位净值", "日增长率", "前20持仓权重", "上涨重仓股数", "上涨重仓股比例", "DIF", "DEA", "MACD柱"]
    cols += ["场外机会分", "盯盘提示", "申购状态", "赎回状态", "手续费", "快照时间"]
    cols += [col for col in out.columns if ("单位净值" in col or "累计净值" in col or col in {"公布日增长率", "公布单位净值", "上一净值", "估算日期", "数据来源"}) and col not in cols]
    return out[[col for col in cols if col in out.columns]]


def latest_otc_snapshot_row(snapshot_df: pd.DataFrame | None, code: str) -> pd.Series | None:
    if snapshot_df is None or snapshot_df.empty or "基金代码" not in snapshot_df.columns:
        return None
    hit = snapshot_df[snapshot_df["基金代码"].astype(str).str.zfill(6) == clean_code(code)]
    return hit.iloc[0] if not hit.empty else None


def snapshot_to_open_fund_row(row: pd.Series | None) -> pd.Series | None:
    if row is None:
        return None
    return pd.Series(
        {
            "基金代码": str(row.get("基金代码", "")).zfill(6),
            "基金简称": row.get("基金简称", ""),
            "单位净值": row.get("最新单位净值", np.nan),
            "日增长率": row.get("日增长率", np.nan),
            "申购状态": row.get("申购状态", "-"),
            "赎回状态": row.get("赎回状态", "-"),
            "手续费": row.get("手续费", "-"),
        }
    )


def snapshot_to_estimation_row(row: pd.Series | None) -> pd.Series | None:
    if row is None:
        return None
    return pd.Series(
        {
            "基金代码": str(row.get("基金代码", "")).zfill(6),
            "基金名称": row.get("基金简称", ""),
            "估算净值": row.get("估算净值", np.nan),
            "估算涨幅": row.get("估算涨幅", np.nan),
            "估算偏差": row.get("估算偏差", np.nan),
            "估算日期": row.get("快照时间", ""),
        }
    )


def filter_otc_snapshot_table(snapshot_df: pd.DataFrame | None, codes: list[str]) -> pd.DataFrame:
    codes = normalize_code_list(codes)
    if snapshot_df is None or snapshot_df.empty or "基金代码" not in snapshot_df.columns:
        if not codes:
            return pd.DataFrame()
        return pd.DataFrame({"基金代码": codes, "基金简称": ["待后台刷新"] * len(codes), "动作": ["待刷新"] * len(codes)})
    df = snapshot_df.copy()
    df["基金代码"] = df["基金代码"].astype(str).str.zfill(6)
    if codes:
        df = df[df["基金代码"].isin(codes)].copy()
        missing = [code for code in codes if code not in set(df["基金代码"].tolist())]
        if missing:
            df = pd.concat(
                [df, pd.DataFrame({"基金代码": missing, "基金简称": ["待后台刷新"] * len(missing), "动作": ["待刷新"] * len(missing)})],
                ignore_index=True,
            )
    if "场外短线评分" in df.columns:
        df["场外短线评分"] = numeric_series(df["场外短线评分"])
        df = df.sort_values("场外短线评分", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def otc_watch_rank_label(row: pd.Series | dict[str, Any]) -> str:
    code = str(row.get("基金代码", "")).zfill(6)
    name = str(row.get("基金简称") or row.get("基金名称") or "-")
    score = to_num(row.get("场外短线评分", row.get("场外机会分", np.nan)))
    action = str(row.get("动作") or row.get("盯盘提示") or "")
    score_text = f"{score:.1f}" if pd.notna(score) else "-"
    return " | ".join([part for part in [code, name, score_text, action] if part])


def open_fund_model_from_snapshot(row: pd.Series | None, nav_df: pd.DataFrame | None, market_env: dict[str, Any]) -> dict[str, Any]:
    if row is None:
        factor_scores = {"趋势分": 50.0, "净值动能分": 50.0, "持仓穿透分": 50.0, "同类热度分": 50.0, "风险控制分": 50.0}
        return {
            "price_df": pd.DataFrame(),
            "factor_scores": factor_scores,
            "total_score": 50.0,
            "action": "数据不足",
            "action_tone": "warn",
            "sector_info": {"sector": "后台快照缺失", "note": "当前基金还没有后台快照"},
            "sector_name": None,
            "market_env": market_env,
            "positives": [],
            "negatives": ["当前基金还没有后台快照，可先加入场外自选并等待后台刷新。"],
            "raw": {},
        }

    price_df = otc_nav_to_price_df(nav_df)
    factor_scores = {
        "趋势分": to_num(row.get("趋势分")),
        "净值动能分": to_num(row.get("净值动能分")),
        "持仓穿透分": to_num(row.get("持仓穿透分")),
        "同类热度分": to_num(row.get("同类热度分")),
        "风险控制分": to_num(row.get("风险控制分")),
    }
    factor_scores = {key: (50.0 if pd.isna(value) else float(value)) for key, value in factor_scores.items()}
    total = to_num(row.get("场外短线评分"))
    action = str(row.get("动作") or "观察等待")
    if action in {"离场", "禁止追高"}:
        tone = "risk"
    elif action == "减仓":
        tone = "warn"
    elif action == "买入观察":
        tone = "good"
    else:
        tone = "neutral"

    positives: list[str] = []
    negatives: list[str] = []
    snapshot_time = str(row.get("快照时间") or "")
    source = str(row.get("数据来源") or "后台快照")
    positives.append(f"已读取后台快照：{snapshot_time or '最新一批'}")
    positives.append(f"快照来源：{source}")
    if to_num(row.get("近5日")) > 0:
        positives.append(f"近5日净值表现 {to_num(row.get('近5日')):.2f}%")
    if to_num(row.get("重仓估算贡献")) > 0:
        positives.append(f"重仓股实时穿透贡献约 {to_num(row.get('重仓估算贡献')):.2f}%")
    if action in {"减仓", "离场", "禁止追高"}:
        negatives.append(f"后台快照动作提示为：{action}")
    if to_num(row.get("120日回撤")) > 12:
        negatives.append(f"120日回撤约 {to_num(row.get('120日回撤')):.2f}%，位置风险偏高")

    return {
        "price_df": price_df,
        "factor_scores": factor_scores,
        "total_score": clamp(total),
        "action": action,
        "action_tone": tone,
        "sector_info": {"sector": "后台快照极速模式", "note": "详情页使用后台快照，前台已跳过慢速实时接口。"},
        "sector_name": None,
        "market_env": market_env,
        "positives": positives,
        "negatives": negatives,
        "raw": {
            "daily_pct": to_num(row.get("日增长率")),
            "estimate_pct": to_num(row.get("估算涨幅")),
            "estimated_nav": to_num(row.get("估算净值")),
            "estimate_bias": to_num(row.get("估算偏差")),
            "through_nav": through_nav,
            "through_pct": through_pct,
            "ret5": to_num(row.get("近5日")),
            "ret10": to_num(row.get("近10日")),
            "ret20": to_num(row.get("近20日")),
            "peer_rank": to_num(row.get("同类热度分")),
            "achievement_rank": np.nan,
            "stock_position": np.nan,
            "top_weight": to_num(row.get("前20持仓权重")),
            "holding_contribution": to_num(row.get("重仓估算贡献")),
            "holding_up_ratio": to_num(row.get("上涨重仓股比例")),
            "space_to_high": np.nan,
            "drawdown120": to_num(row.get("120日回撤")),
            "volatility20": np.nan,
            "max_drawdown": np.nan,
        },
    }


def build_holding_impact_table(holdings_df: pd.DataFrame | None, a_spot: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(), np.nan
    top = holdings_df.head(20).copy()
    if "股票代码" not in top.columns:
        return top, np.nan
    top["股票代码"] = top["股票代码"].astype(str).str.zfill(6)
    if "占净值比例" in top.columns:
        top["占净值比例"] = numeric_series(top["占净值比例"])

    if a_spot is not None and not a_spot.empty and "代码" in a_spot.columns and "涨跌幅" in a_spot.columns:
        spot = a_spot.copy()
        spot["股票代码"] = spot["代码"].astype(str).str.zfill(6)
        keep_cols = ["股票代码", "最新价", "涨跌幅", "成交额", "主力净流入-净额"]
        keep_cols = [col for col in keep_cols if col in spot.columns]
        for col in ["最新价", "涨跌幅", "成交额", "主力净流入-净额"]:
            if col in spot.columns:
                spot[col] = numeric_series(spot[col])
        top = top.merge(spot[keep_cols], on="股票代码", how="left")
        if "占净值比例" in top.columns and "涨跌幅" in top.columns:
            top["估算贡献"] = top["占净值比例"] * top["涨跌幅"] / 100
            total = top["估算贡献"].dropna().sum()
        else:
            total = np.nan
    else:
        total = np.nan
    return top, total


def nav_trend_chart(nav_df: pd.DataFrame | None, title: str) -> go.Figure:
    if nav_df is None or nav_df.empty:
        return go.Figure()
    df = nav_df.tail(520).copy()
    date_col = "净值日期" if "净值日期" in df.columns else df.columns[0]
    value_cols = [col for col in df.columns if col != date_col]
    fig = go.Figure()
    for col in value_cols[:2]:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[col], mode="lines", name=col, line=dict(width=2)))
    fig.update_layout(title=title, height=360, margin=dict(l=20, r=20, t=45, b=20), legend=dict(orientation="h"))
    return fig


def otc_nav_analysis_chart(price_df: pd.DataFrame, title: str) -> go.Figure:
    if price_df is None or price_df.empty:
        return go.Figure()
    df = add_indicators(price_df).tail(220).copy()
    df["drawdown"] = df["close"] / df["close"].cummax() - 1
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.48, 0.17, 0.20, 0.15],
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="单位净值", line=dict(color="#1f77b4", width=2.2)), row=1, col=1)
    for col, color in [("ma5", "#f0a202"), ("ma20", "#7b61ff"), ("ma60", "#4d4d4d")]:
        if col in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col.upper(), line=dict(color=color, width=1.4)), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["pct"],
            name="单日涨跌%",
            marker_color=np.where(df["pct"] >= 0, "#cf3f35", "#16845b"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["macd_hist"],
            name="MACD柱",
            marker_color=np.where(df["macd_hist"] >= 0, "#cf3f35", "#16845b"),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], mode="lines", name="DIF", line=dict(color="#2f80ed", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], mode="lines", name="DEA", line=dict(color="#f0a202", width=1.3)), row=3, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"] * 100,
            mode="lines",
            name="净值回撤%",
            fill="tozeroy",
            line=dict(color="#8a5a00", width=1.5),
        ),
        row=4,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=720,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="涨跌%", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="回撤%", row=4, col=1)
    return fig


def factor_bar(scores: dict[str, float]) -> go.Figure:
    df = pd.DataFrame({"维度": list(scores.keys()), "分数": [round(v, 1) for v in scores.values()]})
    fig = px.bar(df, x="分数", y="维度", orientation="h", text="分数", range_x=[0, 100], color="分数", color_continuous_scale=["#bd2d28", "#e4b751", "#2f8f5b"])
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def radar_chart(scores: dict[str, float], name: str = "ETF评分") -> go.Figure:
    labels = list(scores.keys())
    values = [round(v, 1) for v in scores.values()]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself", name=name, line_color="#1f77b4"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def candlestick_chart(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df.tail(180).copy()
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.58, 0.22, 0.20],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )
    fig.add_trace(
        go.Candlestick(
            x=plot_df["date"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="K线",
            increasing_line_color="#cf3f35",
            decreasing_line_color="#16845b",
        ),
        row=1,
        col=1,
    )
    ma_colors = {"ma5": "#f0a202", "ma10": "#2f80ed", "ma20": "#7b61ff", "ma60": "#4d4d4d"}
    for col, color in ma_colors.items():
        if col in plot_df:
            fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df[col], mode="lines", name=col.upper(), line=dict(width=1.6, color=color)), row=1, col=1)
    bar_colors = np.where(plot_df["close"] >= plot_df["open"], "#cf3f35", "#16845b")
    fig.add_trace(go.Bar(x=plot_df["date"], y=plot_df["amount"] / 1e8, marker_color=bar_colors, name="成交额(亿)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["amount_ma5"] / 1e8, mode="lines", name="5日均额", line=dict(color="#555", width=1.3)), row=2, col=1)
    fig.add_trace(go.Bar(x=plot_df["date"], y=plot_df["macd_hist"], marker_color=np.where(plot_df["macd_hist"] >= 0, "#cf3f35", "#16845b"), name="MACD柱"), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["macd"], mode="lines", name="DIF", line=dict(color="#2f80ed", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["macd_signal"], mode="lines", name="DEA", line=dict(color="#f0a202", width=1.2)), row=3, col=1)
    fig.update_layout(
        title=title,
        height=680,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交额(亿)", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def industry_heat_chart(sector_df: pd.DataFrame | None) -> go.Figure:
    if sector_df is None or sector_df.empty:
        return go.Figure()
    top = sector_df.sort_values("pct_chg", ascending=False).head(20).copy()
    fig = px.bar(
        top.sort_values("pct_chg"),
        x="pct_chg",
        y="sector",
        orientation="h",
        color="net_inflow",
        color_continuous_scale=["#16845b", "#f3d17c", "#cf3f35"],
        hover_data=["amount", "up_ratio", "leader", "leader_pct"],
        labels={"pct_chg": "涨跌幅%", "sector": "", "net_inflow": "净流入"},
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def index_chart(index_df: pd.DataFrame | None, title: str) -> go.Figure:
    if index_df is None or index_df.empty:
        return go.Figure()
    df = add_indicators(index_df).tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="收盘", line=dict(color="#1f77b4", width=2)))
    for col, color in [("ma20", "#f0a202"), ("ma60", "#555")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col.upper(), line=dict(color=color, width=1.4)))
    fig.update_layout(title=title, height=330, margin=dict(l=20, r=20, t=45, b=20), legend=dict(orientation="h"))
    return fig


def score_badge(action: str, tone: str) -> str:
    colors = {
        "good": ("#e7f6ed", "#166534"),
        "neutral": ("#edf2f7", "#263238"),
        "warn": ("#fff5d6", "#8a5a00"),
        "risk": ("#fde8e7", "#9f1d1d"),
    }
    bg, fg = colors.get(tone, colors["neutral"])
    return f"<span class='badge' style='background:{bg};color:{fg}'>{action}</span>"


def render_model_references() -> None:
    st.subheader("模型依据与边界")
    st.write(
        "本工具采用“学术可解释因子 + A股短线风控经验”的混合框架：趋势、动量、量价和流动性来自可追溯研究；"
        "主力资金、板块热度、跌破均线后的动作提示属于 A 股短线交易经验规则，需要结合实盘滑点和数据延迟校验。"
    )
    ref_df = pd.DataFrame(MODEL_REFERENCES)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    st.caption("模型不是收益承诺。公开基金持仓通常滞后披露，穿透贡献是估算项；ETF盘口、折溢价和主力资金字段依赖免费行情接口，需关注延迟和字段变化。")


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e6e8eb; border-radius: 8px; padding: 10px 12px; }
    .badge { display:inline-block; padding: 6px 10px; border-radius: 8px; font-weight: 700; }
    .small-note { color: #667085; font-size: 0.86rem; }
    .source-ok { color: #166534; }
    .source-bad { color: #9f1d1d; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("数据库")
    fund_mode = st.radio(
        "分析类型",
        ["场内ETF", "场外基金"],
        index=0 if st.session_state.get("fund_mode", "场内ETF") == "场内ETF" else 1,
        horizontal=True,
    )
    st.session_state["fund_mode"] = fund_mode
    db_read_mode = st.toggle("优先读取数据库快照", value=st.session_state.get("db_read_mode", False))
    st.session_state["db_read_mode"] = db_read_mode
    otc_snapshot_mode = st.toggle("场外自选优先读后台快照", value=st.session_state.get("otc_snapshot_mode", True))
    st.session_state["otc_snapshot_mode"] = otc_snapshot_mode
    use_db_watchlist = st.toggle("自选池保存到数据库", value=st.session_state.get("use_db_watchlist", False))
    st.session_state["use_db_watchlist"] = use_db_watchlist
    if st.button("同步实时行情入库", use_container_width=True):
        try:
            result = collect_market_snapshot()
            st.cache_data.clear()
            st.success(f"已入库 ETF {result['etf']['rows']} 行")
        except Exception as exc:  # noqa: BLE001
            st.error(f"同步失败：{type(exc).__name__}: {exc}")
    if st.button("同步场外自选快照入库", use_container_width=True):
        try:
            result = collect_otc_watch_snapshot()
            st.cache_data.clear()
            st.success(f"已入库场外自选 {result['rows']} 只，用时 {result.get('elapsed_seconds', 0):.1f} 秒")
        except Exception as exc:  # noqa: BLE001
            st.error(f"场外同步失败：{type(exc).__name__}: {exc}")
    with st.expander("数据库状态", expanded=False):
        try:
            summary = database_summary()
            st.write(f"{summary['backend']} ｜ {summary['url']}")
            st.json({k: v for k, v in summary.items() if k not in {"backend", "url"}})
        except Exception as exc:  # noqa: BLE001
            st.warning(f"数据库暂不可用：{type(exc).__name__}: {exc}")

if fund_mode == "场外基金" and otc_snapshot_mode:
    etf_spot, spot_status = None, data_status("ETF实时行情-东方财富", True, "场外后台快照模式已跳过")
elif db_read_mode:
    etf_spot, spot_status = get_etf_spot_from_db()
    if etf_spot is None:
        live_spot, live_status = get_etf_spot()
        etf_spot, spot_status = live_spot, {"ok": live_status.get("ok"), "label": "数据库为空，已回退实时ETF行情", "detail": live_status.get("detail")}
else:
    etf_spot, spot_status = get_etf_spot()

if "selected_etf_code" not in st.session_state:
    st.session_state["selected_etf_code"] = "510300"
if "selected_otc_code" not in st.session_state:
    st.session_state["selected_otc_code"] = "110022"
watchlist_codes = load_watchlist()
otc_watchlist_codes = load_otc_watchlist()
if fund_mode == "场外基金" and otc_snapshot_mode:
    otc_watch_snapshot_df, otc_watch_snapshot_status = get_otc_watch_snapshot_from_db()
else:
    otc_watch_snapshot_df, otc_watch_snapshot_status = None, data_status("场外自选后台快照", True, "未启用或非场外模式")
if fund_mode == "场外基金":
    if otc_snapshot_mode and otc_watch_snapshot_df is not None and not otc_watch_snapshot_df.empty:
        with st.spinner("正在读取场外基金后台快照..."):
            open_fund_daily = None
            open_fund_daily_status = data_status("开放式基金净值-东方财富/天天基金", True, "已使用后台快照，前台跳过全量净值表")
            fund_names, fund_names_status = get_fund_names_from_collector_cache()
            open_fund_estimation = None
            open_fund_estimation_status = data_status("场外基金盘中估值-东方财富", True, "已使用后台快照，前台跳过全量估算表")
    else:
        with st.spinner("正在读取场外基金列表..."):
            open_fund_daily, open_fund_daily_status = get_open_fund_daily()
            fund_names, fund_names_status = get_fund_names()
            open_fund_estimation, open_fund_estimation_status = get_open_fund_estimation()
else:
    open_fund_daily, open_fund_daily_status = None, data_status("开放式基金净值-东方财富/天天基金", True, "场内模式未读取")
    fund_names, fund_names_status = None, data_status("全部基金名称-东方财富/天天基金", True, "场内模式未读取")
    open_fund_estimation, open_fund_estimation_status = None, data_status("场外基金盘中估值-东方财富", True, "场内模式未读取")


with st.sidebar:
    st.header("参数")
    code = clean_code(st.session_state.get("selected_etf_code", "510300"))
    otc_code = clean_code(st.session_state.get("selected_otc_code", "110022"))

    if fund_mode == "场内ETF":
        st.subheader("ETF实时查询")
        search_query = st.text_input("搜索代码/名称/主题", value="", placeholder="如 510300、证券、机器人")
        current_code = clean_code(st.session_state.get("selected_etf_code", "510300"))
        search_options = build_etf_search_options(etf_spot, search_query, current_code)
        selected_index = 0
        for idx, option in enumerate(search_options):
            if extract_code_from_label(option) == current_code:
                selected_index = idx
                break
        selected_option = st.selectbox("ETF列表", search_options, index=selected_index)
        selected_code = extract_code_from_label(selected_option) or current_code
        manual_code = st.text_input("手动代码", value="", placeholder="可选：直接输入6位代码")
        code = clean_code(manual_code) if maybe_clean_code(manual_code) else selected_code
        st.session_state["selected_etf_code"] = code

        add_col, refresh_col = st.columns(2)
        with add_col:
            if st.button("加入自选", use_container_width=True):
                watchlist_codes = save_watchlist([*watchlist_codes, code])
                st.rerun()
        with refresh_col:
            if st.button("刷新实时数据", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        st.subheader("场内自选池")
        watchlist_label_options = [etf_label_for_code(etf_spot, item) for item in watchlist_codes]
        if watchlist_label_options:
            focus_label = st.selectbox("快速切换自选", watchlist_label_options)
            if st.button("分析选中自选", use_container_width=True):
                st.session_state["selected_etf_code"] = extract_code_from_label(focus_label) or code
                st.rerun()
        watchlist_text = st.text_area("批量编辑场内代码", value=" ".join(watchlist_codes), height=72, help="用空格、逗号或换行分隔代码。")
        save_col, clear_col = st.columns(2)
        with save_col:
            if st.button("保存场内自选", use_container_width=True):
                watchlist_codes = save_watchlist(watchlist_text)
                st.rerun()
        with clear_col:
            if st.button("清空场内自选", use_container_width=True):
                watchlist_codes = save_watchlist([])
                st.rerun()
    else:
        st.subheader("场外基金查询")
        otc_query = st.text_input("搜索场外基金", value="", placeholder="如 消费、白酒、110022、E方达")
        current_otc_code = clean_code(st.session_state.get("selected_otc_code", "110022"))
        otc_options = build_open_fund_search_options(open_fund_daily, fund_names, otc_query, current_otc_code)
        otc_selected_index = 0
        for idx, option in enumerate(otc_options):
            if extract_code_from_label(option) == current_otc_code:
                otc_selected_index = idx
                break
        otc_option = st.selectbox("场外基金列表", otc_options, index=otc_selected_index)
        otc_manual_code = st.text_input("场外基金手动代码", value="", placeholder="可选：直接输入6位代码")
        otc_code = clean_code(otc_manual_code) if maybe_clean_code(otc_manual_code) else (extract_code_from_label(otc_option) or current_otc_code)
        st.session_state["selected_otc_code"] = otc_code

        otc_add_col, otc_focus_col = st.columns(2)
        with otc_add_col:
            if st.button("加入场外自选", use_container_width=True):
                otc_watchlist_codes = save_otc_watchlist([*otc_watchlist_codes, otc_code])
                st.rerun()
        with otc_focus_col:
            if st.button("分析场外基金", use_container_width=True):
                st.session_state["selected_otc_code"] = otc_code
                st.rerun()

        otc_watchlist_text = st.text_area("场外自选代码", value=" ".join(otc_watchlist_codes), height=60)
        otc_watchlist_label_options = [open_fund_label_for_code(open_fund_daily, fund_names, item) for item in otc_watchlist_codes]
        if otc_watchlist_label_options:
            current_focus_code = clean_code(st.session_state.get("selected_otc_code", otc_code))
            focus_index = 0
            for idx, label in enumerate(otc_watchlist_label_options):
                if extract_code_from_label(label) == current_focus_code:
                    focus_index = idx
                    break
            current_focus_label = otc_watchlist_label_options[focus_index]
            if (
                st.session_state.get("sidebar_otc_watch_focus") not in otc_watchlist_label_options
                or extract_code_from_label(st.session_state.get("sidebar_otc_watch_focus", "")) != current_focus_code
            ):
                st.session_state["sidebar_otc_watch_focus"] = current_focus_label
            focus_otc_label = st.selectbox("点击自选基金查看分析", otc_watchlist_label_options, index=focus_index, key="sidebar_otc_watch_focus")
            focus_otc_code = extract_code_from_label(focus_otc_label)
            if focus_otc_code and focus_otc_code != current_focus_code:
                st.session_state["selected_otc_code"] = focus_otc_code
                st.rerun()
        otc_save_col, otc_clear_col = st.columns(2)
        with otc_save_col:
            if st.button("保存场外自选", use_container_width=True):
                otc_watchlist_codes = save_otc_watchlist(otc_watchlist_text)
                st.rerun()
        with otc_clear_col:
            if st.button("清空场外自选", use_container_width=True):
                otc_watchlist_codes = save_otc_watchlist([])
                st.rerun()

    index_label = st.selectbox("对比指数", list(INDEX_OPTIONS.keys()), index=0)
    index_symbol = INDEX_OPTIONS[index_label]
    custom_sector = st.text_input("板块关键词", value="", placeholder="可选：如 证券、半导体、机器人")
    lookback = st.slider("K线回看天数", min_value=120, max_value=900, value=420, step=30)
    leaderboard_size = st.slider("实时机会榜数量", min_value=10, max_value=80, value=30, step=10)
    st.caption("数据会按短 TTL 缓存。点击右上角 Rerun 可刷新。")


end_dt = date.today()
start_dt = end_dt - timedelta(days=int(lookback * 1.7))
start_str = start_dt.strftime("%Y%m%d")
end_str = end_dt.strftime("%Y%m%d")
otc_snapshot_focus_row = latest_otc_snapshot_row(otc_watch_snapshot_df, otc_code)
use_otc_fast_snapshot = fund_mode == "场外基金" and otc_snapshot_mode and otc_snapshot_focus_row is not None

with st.spinner("正在读取实时数据..."):
    index_df, index_statuses = get_index_daily(index_symbol, start_str, end_str)
    if fund_mode == "场内ETF":
        daily_df, daily_statuses = get_etf_daily(code, start_str, end_str)
        overview_df, overview_status = get_fund_overview(code)
        nav_df, nav_status = get_fund_nav(code, start_str, end_str)
        holdings_df, holdings_statuses = get_holdings(code)
        otc_nav_df, otc_nav_status = None, data_status("场外基金净值", True, "场内模式未读取")
        otc_basic_df, otc_basic_status = None, data_status("场外基金基本信息", True, "场内模式未读取")
        otc_achievement_df, otc_achievement_status = None, data_status("场外基金业绩", True, "场内模式未读取")
        otc_asset_df, otc_asset_status = None, data_status("场外基金资产配置", True, "场内模式未读取")
        otc_holdings_df, otc_holdings_statuses = None, [data_status("场外基金股票持仓", True, "场内模式未读取")]
    else:
        daily_df, daily_statuses = None, [data_status("ETF日K", True, "场外模式未读取")]
        overview_df, overview_status = None, data_status("ETF基金概况", True, "场外模式未读取")
        nav_df, nav_status = None, data_status("ETF净值", True, "场外模式未读取")
        holdings_df, holdings_statuses = None, [data_status("ETF成分持仓", True, "场外模式未读取")]
        if use_otc_fast_snapshot:
            otc_nav_df, otc_nav_status = get_open_fund_nav_from_collector_cache(otc_code)
            otc_basic_df, otc_basic_status = None, data_status("场外基金基本信息", True, "后台快照极速模式已跳过")
            otc_achievement_df, otc_achievement_status = None, data_status("场外基金业绩", True, "后台快照极速模式已跳过")
            otc_asset_df, otc_asset_status = None, data_status("场外基金资产配置", True, "后台快照极速模式已跳过")
            otc_holdings_df, otc_holdings_statuses = None, [data_status("场外基金股票持仓", True, "后台快照极速模式已跳过")]
        else:
            otc_nav_df, otc_nav_status = get_open_fund_nav_trend(otc_code)
            otc_basic_df, otc_basic_status = get_open_fund_basic(otc_code)
            otc_achievement_df, otc_achievement_status = get_open_fund_achievement(otc_code)
            otc_asset_df, otc_asset_status = get_open_fund_asset_allocation(otc_code)
            otc_holdings_df, otc_holdings_statuses = get_open_fund_holdings(otc_code)
    if db_read_mode:
        sector_df, sector_status = get_sector_summary_from_db()
        if sector_df is None:
            live_sector_df, live_sector_status = get_sector_summary()
            sector_df, sector_status = live_sector_df, {"ok": live_sector_status.get("ok"), "label": "数据库为空，已回退实时行业热度", "detail": live_sector_status.get("detail")}
    elif use_otc_fast_snapshot:
        sector_df, sector_status = get_sector_summary_from_db()
        if sector_df is None:
            sector_df, sector_status = None, data_status("行业热度-数据库", True, "后台快照极速模式已跳过实时行业接口")
    else:
        sector_df, sector_status = get_sector_summary()
    if use_otc_fast_snapshot:
        a_spot, a_spot_status = None, data_status("A股实时行情-东方财富", True, "后台快照极速模式已跳过")
    else:
        a_spot, a_spot_status = get_a_spot()


if fund_mode == "场内ETF" and (daily_df is None or daily_df.empty):
    st.error("没有拿到该 ETF 的可用 K 线。请检查代码，或稍后再试数据源。")
    with st.expander("数据源状态"):
        for status in [spot_status, *daily_statuses, *index_statuses, overview_status, nav_status, *holdings_statuses, sector_status, a_spot_status]:
            klass = "source-ok" if status.get("ok") else "source-bad"
            state = "OK" if status.get("ok") else "FAIL"
            st.markdown(f"<span class='{klass}'>{state}</span> {status.get('label')}: {status.get('detail')}", unsafe_allow_html=True)
    st.stop()


market_env = compute_market_env(index_df, a_spot, etf_spot)
spot_row = latest_spot_row(etf_spot, code)
overview = overview_fields(overview_df)
etf_name = str(spot_row.get("名称")) if spot_row is not None and "名称" in spot_row.index else overview.get("基金简称", code)
track_target = overview.get("跟踪标的", "")
sector_name = infer_sector_name(etf_name, track_target, custom_sector, sector_df)
if daily_df is not None and not daily_df.empty:
    daily_df = add_indicators(daily_df).tail(lookback).reset_index(drop=True)
    model = score_model(daily_df, index_df, spot_row, etf_spot, sector_df, sector_name, market_env)
else:
    daily_df = pd.DataFrame()
    model = {
        "latest": {"date": date.today(), "close": np.nan, "pct": np.nan},
        "raw": {},
        "factor_scores": {"趋势分": 50.0, "量能分": 50.0, "资金分": 50.0, "板块热度分": 50.0, "风险控制分": 50.0},
        "total_score": 50.0,
        "action": "数据不足",
        "action_tone": "warn",
        "positives": [],
        "negatives": ["ETF K线数据不可用，场内评分暂按中性处理"],
        "sector_info": {},
    }
latest = model["latest"]
raw = model["raw"]
factor_scores = model["factor_scores"]
leaderboard = build_etf_leaderboard(etf_spot, leaderboard_size)
watchlist_table = build_realtime_etf_table(etf_spot, codes=watchlist_codes)
etf_holding_impact, etf_estimated_contribution = build_holding_impact_table(holdings_df, a_spot)
if otc_snapshot_mode and otc_watch_snapshot_df is not None and not otc_watch_snapshot_df.empty:
    otc_watchlist_table = filter_otc_snapshot_table(otc_watch_snapshot_df, otc_watchlist_codes)
else:
    otc_watchlist_table = build_open_fund_watch_table(open_fund_daily, fund_names, otc_watchlist_codes, open_fund_estimation)
otc_row = latest_open_fund_row(open_fund_daily, otc_code)
if otc_row is None:
    otc_row = snapshot_to_open_fund_row(otc_snapshot_focus_row)
otc_estimation_row = latest_open_fund_estimation_row(open_fund_estimation, otc_code)
if otc_estimation_row is None:
    otc_estimation_row = snapshot_to_estimation_row(otc_snapshot_focus_row)
otc_name = open_fund_name_for_code(open_fund_daily, fund_names, otc_snapshot_focus_row, otc_estimation_row, otc_code)
otc_holding_impact, otc_estimated_contribution = build_holding_impact_table(otc_holdings_df, a_spot)
otc_sector_name = infer_sector_name(otc_name, "", custom_sector, sector_df)
if use_otc_fast_snapshot:
    otc_model = open_fund_model_from_snapshot(otc_snapshot_focus_row, otc_nav_df, market_env)
    otc_estimated_contribution = to_num(otc_snapshot_focus_row.get("重仓估算贡献")) if otc_snapshot_focus_row is not None else np.nan
else:
    otc_model = score_open_fund_model(
        otc_nav_df,
        index_df,
        open_fund_daily,
        open_fund_estimation,
        otc_row,
        otc_estimation_row,
        otc_achievement_df,
        otc_asset_df,
        otc_holding_impact,
        sector_df,
        otc_sector_name,
        market_env,
    )
otc_factor_scores = otc_model["factor_scores"]
otc_raw = otc_model["raw"]


if fund_mode == "场外基金":
    st.title("A股场外基金短线机会评分台")
    st.caption("场外基金页只显示开放式基金体系：净值趋势、阶段业绩、公开持仓穿透、同类热度和风险控制。公开数据不包含支付宝/微信账户持仓。")
    if use_otc_fast_snapshot:
        st.success("已启用后台快照极速模式：当前基金详情读取 otc_watch_snapshot，前台跳过慢速实时接口。")

    title_left, title_mid, title_right = st.columns([2.2, 1.1, 1.2])
    with title_left:
        st.subheader(f"{otc_code}  {otc_name}")
        nav_date = "-"
        if otc_model["price_df"] is not None and not otc_model["price_df"].empty:
            nav_date = pd.to_datetime(otc_model["price_df"]["date"].iloc[-1]).date()
        st.markdown(f"<span class='small-note'>最新净值日：{nav_date} ｜ 市场环境：{market_env['regime']}</span>", unsafe_allow_html=True)
    with title_mid:
        st.metric("场外短线评分", f"{otc_model['total_score']:.1f}", help="趋势25%、净值动能20%、持仓穿透20%、同类热度20%、风险15%")
    with title_right:
        st.markdown("状态")
        st.markdown(score_badge(otc_model["action"], otc_model["action_tone"]), unsafe_allow_html=True)

    otc_header_cols = st.columns(6)
    nav_col = next((col for col in otc_row.index if "单位净值" in str(col)), None) if otc_row is not None else None
    latest_unit_nav = to_num(otc_row.get(nav_col)) if otc_row is not None and nav_col else np.nan
    if pd.isna(latest_unit_nav) and otc_model["price_df"] is not None and not otc_model["price_df"].empty:
        latest_unit_nav = to_num(otc_model["price_df"]["close"].iloc[-1])
    daily_pct_value = to_num(otc_row.get("日增长率")) if otc_row is not None else to_num(otc_raw.get("daily_pct"))
    through_nav_value = to_num(otc_raw.get("through_nav"))
    if pd.isna(through_nav_value):
        through_nav_value = to_num(otc_raw.get("estimated_nav"))
    through_pct_value = to_num(otc_raw.get("through_pct"))
    if pd.isna(through_pct_value):
        through_pct_value = to_num(otc_raw.get("estimate_pct"))
    otc_header_cols[0].metric("最新单位净值", f"{latest_unit_nav:.4f}" if pd.notna(latest_unit_nav) else "-")
    otc_header_cols[1].metric("日增长率", pct_text(daily_pct_value))
    otc_header_cols[2].metric("实时穿透净值", f"{through_nav_value:.4f}" if pd.notna(through_nav_value) else "-")
    otc_header_cols[3].metric("实时穿透涨幅", pct_text(through_pct_value), delta=f"估算偏差 {pct_text(to_num(otc_raw.get('estimate_bias')))}")
    otc_header_cols[4].metric("同类/热度分位", pct_text(to_num(otc_raw.get("peer_rank")), 1))
    otc_header_cols[5].metric("申/赎状态", f"{otc_row.get('申购状态', '-') if otc_row is not None else '-'} / {otc_row.get('赎回状态', '-') if otc_row is not None else '-'}")

    otc_tabs = st.tabs(["机会评分", "净值与动量", "资金与穿透", "板块与市场", "持仓与资产", "自选池/查询", "模型"])

    with otc_tabs[0]:
        factor_left, factor_right = st.columns([1.15, 1])
        with factor_left:
            st.plotly_chart(factor_bar(otc_factor_scores), use_container_width=True)
        with factor_right:
            st.plotly_chart(radar_chart(otc_factor_scores, "场外基金评分"), use_container_width=True)

        reason_left, reason_right = st.columns(2)
        with reason_left:
            st.markdown("**正向证据**")
            if otc_model["positives"]:
                for item in otc_model["positives"]:
                    st.write(f"- {item}")
            else:
                st.write("暂无明显正向确认。")
        with reason_right:
            st.markdown("**风险信号**")
            if otc_model["negatives"]:
                for item in otc_model["negatives"]:
                    st.write(f"- {item}")
            else:
                st.write("暂无明显负向信号。")

        otc_detail_cols = st.columns(6)
        otc_detail_cols[0].metric("近5日净值", pct_text(to_num(otc_raw.get("ret5"))))
        otc_detail_cols[1].metric("近20日净值", pct_text(to_num(otc_raw.get("ret20"))))
        otc_detail_cols[2].metric("实时穿透涨幅", pct_text(to_num(otc_raw.get("through_pct"))))
        otc_detail_cols[3].metric("重仓估算贡献", pct_text(to_num(otc_raw.get("holding_contribution"))))
        otc_detail_cols[4].metric("前20持仓权重", pct_text(to_num(otc_raw.get("top_weight"))))
        otc_detail_cols[5].metric("120日回撤", pct_text(to_num(otc_raw.get("drawdown120"))))

        st.subheader("场外自选基金短线操作评分排行")
        if otc_watchlist_table.empty:
            st.info("场外自选池暂无排行。可先在左侧加入基金，再点击“同步场外自选快照入库”。")
        else:
            rank_df = otc_watchlist_table.copy()
            if "场外短线评分" in rank_df.columns:
                rank_df["场外短线评分"] = numeric_series(rank_df["场外短线评分"])
                chart_df = rank_df.dropna(subset=["场外短线评分"]).copy()
                if not chart_df.empty:
                    name_col = "基金简称" if "基金简称" in chart_df.columns else "基金代码"
                    fig = px.bar(
                        chart_df.sort_values("场外短线评分"),
                        x="场外短线评分",
                        y=name_col,
                        color="动作" if "动作" in chart_df.columns else None,
                        orientation="h",
                        hover_data=[col for col in ["基金代码", "估算涨幅", "实时穿透涨幅", "重仓估算贡献", "近5日", "120日回撤", "快照时间"] if col in chart_df.columns],
                    )
                    fig.update_layout(height=max(260, min(520, 42 * len(chart_df) + 120)), margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            rank_labels = [otc_watch_rank_label(row) for _, row in rank_df.iterrows()]
            if rank_labels:
                current_rank_index = 0
                for idx, label in enumerate(rank_labels):
                    if extract_code_from_label(label) == otc_code:
                        current_rank_index = idx
                        break
                chosen_rank_label = st.selectbox("点击排行基金查看分析", rank_labels, index=current_rank_index, key="otc_rank_click_select")
                chosen_rank_code = extract_code_from_label(chosen_rank_label)
                if chosen_rank_code and chosen_rank_code != otc_code:
                    st.session_state["selected_otc_code"] = chosen_rank_code
                    st.rerun()
            st.dataframe(format_open_fund_display(rank_df), use_container_width=True, hide_index=True)

    with otc_tabs[1]:
        if not otc_model["price_df"].empty:
            st.plotly_chart(otc_nav_analysis_chart(otc_model["price_df"], f"{otc_code} {otc_name}：净值趋势 / 涨跌 / MACD / 回撤"), use_container_width=True)
            otc_ind = add_indicators(otc_model["price_df"])
            otc_last = otc_ind.iloc[-1]
            tech_cols = st.columns(10)
            tech_cols[0].metric("MA5", f"{to_num(otc_last.get('ma5')):.4f}")
            tech_cols[1].metric("MA10", f"{to_num(otc_last.get('ma10')):.4f}")
            tech_cols[2].metric("MA20", f"{to_num(otc_last.get('ma20')):.4f}")
            tech_cols[3].metric("MA60", f"{to_num(otc_last.get('ma60')):.4f}")
            tech_cols[4].metric("近5日", pct_text(to_num(otc_raw.get("ret5"))))
            tech_cols[5].metric("近20日", pct_text(to_num(otc_raw.get("ret20"))))
            tech_cols[6].metric("DIF", f"{to_num(otc_last.get('macd')):.4f}")
            tech_cols[7].metric("DEA", f"{to_num(otc_last.get('macd_signal')):.4f}")
            tech_cols[8].metric("MACD柱", f"{to_num(otc_last.get('macd_hist')):.4f}")
            tech_cols[9].metric("120日回撤", pct_text(to_num(otc_raw.get("drawdown120"))))
        else:
            st.plotly_chart(nav_trend_chart(otc_nav_df, f"{otc_code} {otc_name}：单位净值走势"), use_container_width=True)
        st.caption("净值动量会把后台快照里的实时穿透净值追加为最新点；DIF、DEA、MACD柱因此会反映当天重仓股穿透估算。")

    with otc_tabs[2]:
        st.subheader("重仓股实时穿透")
        impact_cols = st.columns(5)
        impact_cols[0].metric("重仓股估算贡献", pct_text(otc_estimated_contribution))
        impact_cols[1].metric("实时穿透涨幅", pct_text(to_num(otc_raw.get("through_pct"))))
        impact_cols[2].metric("实时穿透净值", f"{to_num(otc_raw.get('through_nav')):.4f}" if pd.notna(to_num(otc_raw.get("through_nav"))) else "-")
        if otc_holding_impact is not None and not otc_holding_impact.empty and "占净值比例" in otc_holding_impact:
            impact_cols[3].metric("前20持仓权重", pct_text(otc_holding_impact["占净值比例"].dropna().sum()))
        else:
            impact_cols[3].metric("前20持仓权重", pct_text(to_num(otc_raw.get("top_weight"))))
        if otc_holding_impact is not None and not otc_holding_impact.empty and "涨跌幅" in otc_holding_impact:
            impact_cols[4].metric("上涨重仓股数", f"{int((otc_holding_impact['涨跌幅'] > 0).sum())}")
        else:
            impact_cols[4].metric("上涨重仓比例", pct_text(to_num(otc_raw.get("holding_up_ratio"))))
        if otc_holding_impact is not None and not otc_holding_impact.empty and "主力净流入-净额" in otc_holding_impact:
            st.metric("重仓主力净额", format_money(numeric_series(otc_holding_impact["主力净流入-净额"]).sum()))
        if otc_holding_impact is not None and not otc_holding_impact.empty:
            display_impact = otc_holding_impact.copy()
            for col in ["占净值比例", "涨跌幅", "估算贡献"]:
                if col in display_impact:
                    display_impact[col] = display_impact[col].map(lambda x: pct_text(x))
            for col in ["成交额", "主力净流入-净额"]:
                if col in display_impact:
                    display_impact[col] = display_impact[col].map(format_money)
            keep_cols = ["股票代码", "股票名称", "占净值比例", "最新价", "涨跌幅", "估算贡献", "成交额", "主力净流入-净额", "季度"]
            st.dataframe(display_impact[[col for col in keep_cols if col in display_impact.columns]], use_container_width=True, hide_index=True)
        else:
            st.write("未展开逐股持仓表；上方指标来自后台快照的重仓股穿透估算。")

        st.subheader("穿透解释")
        st.write("场外基金没有交易所盘口和逐笔成交，资金确认使用公开重仓股实时涨跌、重仓主力净额、上涨重仓股比例和持仓权重集中度作为代理。")

    with otc_tabs[3]:
        left, right = st.columns([1.3, 1])
        with left:
            st.subheader("行业热度")
            st.plotly_chart(industry_heat_chart(sector_df), use_container_width=True)
        with right:
            st.subheader("市场环境")
            st.metric("环境分", f"{market_env['score']:.1f}", delta=market_env["regime"])
            st.write(market_env["index_label"])
            st.write(market_env["breadth_label"])
            st.plotly_chart(index_chart(index_df, index_label), use_container_width=True)
            st.subheader("映射板块")
            sector_info = otc_model.get("sector_info", {})
            if otc_sector_name and sector_info.get("sector") not in {"无板块数据", "未匹配具体行业"}:
                st.metric("板块", otc_sector_name, delta=pct_text(to_num(sector_info.get("pct_chg"))))
                st.write(f"成交额排名：{to_num(sector_info.get('rank_amount')):.0f} ｜ 资金排名：{to_num(sector_info.get('rank_flow')):.0f}")
            else:
                st.write(sector_info.get("note", "未映射到明确板块，按同类日表现和市场环境处理。"))

    with otc_tabs[4]:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("资产配置")
            if otc_asset_df is not None and not otc_asset_df.empty:
                fig = px.pie(otc_asset_df, names="资产类型", values="仓位占比", hole=0.45)
                fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(otc_asset_df, use_container_width=True, hide_index=True)
            else:
                st.write("未取得资产配置。")

            st.subheader("基本信息")
            if otc_basic_df is not None and not otc_basic_df.empty:
                st.dataframe(otc_basic_df, use_container_width=True, hide_index=True)
            else:
                st.write("未取得基本信息。")
        with right:
            st.subheader("阶段业绩")
            if otc_achievement_df is not None and not otc_achievement_df.empty:
                st.dataframe(otc_achievement_df, use_container_width=True, hide_index=True)
            else:
                st.write("未取得阶段业绩。")

    with otc_tabs[5]:
        watch_left, browse_right = st.columns([1, 1.15])
        with watch_left:
            st.subheader("场外基金自选池")
            if otc_watchlist_table.empty:
                st.info("场外基金自选池为空。可在左侧搜索后加入。")
            else:
                if "基金代码" in otc_watchlist_table.columns:
                    otc_watch_labels = [otc_watch_rank_label(row) for _, row in otc_watchlist_table.iterrows()]
                else:
                    otc_watch_labels = [open_fund_label_for_code(open_fund_daily, fund_names, item) for item in otc_watchlist_codes]
                current_watch_index = 0
                for idx, label in enumerate(otc_watch_labels):
                    if extract_code_from_label(label) == otc_code:
                        current_watch_index = idx
                        break
                current_watch_label = otc_watch_labels[current_watch_index]
                if (
                    st.session_state.get("otc_watchlist_click_select") not in otc_watch_labels
                    or extract_code_from_label(st.session_state.get("otc_watchlist_click_select", "")) != otc_code
                ):
                    st.session_state["otc_watchlist_click_select"] = current_watch_label
                chosen_otc_watch = st.selectbox("点击自选基金查看分析", otc_watch_labels, index=current_watch_index, key="otc_watchlist_click_select")
                chosen_otc_code = extract_code_from_label(chosen_otc_watch)
                if chosen_otc_code and chosen_otc_code != otc_code:
                    st.session_state["selected_otc_code"] = chosen_otc_code
                    st.rerun()
                st.dataframe(format_open_fund_display(otc_watchlist_table), use_container_width=True, hide_index=True)
        with browse_right:
            st.subheader("开放式基金查询")
            otc_browse_query = st.text_input("场外基金表内搜索", value="", placeholder="输入基金代码、名称、类型或拼音", key="otc_browse_query")
            otc_browse_options = build_open_fund_search_options(open_fund_daily, fund_names, otc_browse_query, otc_code, limit=300)
            browse_codes = [extract_code_from_label(item) for item in otc_browse_options]
            otc_browse_table = build_open_fund_watch_table(open_fund_daily, fund_names, browse_codes, open_fund_estimation)
            st.dataframe(format_open_fund_display(otc_browse_table), use_container_width=True, hide_index=True)

    with otc_tabs[6]:
        st.subheader("场外基金评分公式")
        st.code("场外基金短线强度 = 趋势分*0.25 + 净值动能分*0.20 + 持仓穿透分*0.20 + 同类热度分*0.20 + 风险控制分*0.15", language="text")
        st.write(
            "减仓：净值跌破20日线，或重仓股估算贡献小于 -0.30% 且上涨重仓股比例偏低，或120日回撤超过12%。"
            "离场：净值跌破60日线，并且同类热度分低于48或持仓穿透分低于45。"
        )
        render_model_references()
        st.subheader("数据源状态")
        statuses = [otc_watch_snapshot_status, open_fund_daily_status, open_fund_estimation_status, fund_names_status, otc_nav_status, otc_basic_status, otc_achievement_status, otc_asset_status, *otc_holdings_statuses, *index_statuses, sector_status, a_spot_status]
        status_df = pd.DataFrame([{"数据源": s.get("label"), "状态": "OK" if s.get("ok") else "FAIL", "说明": s.get("detail")} for s in statuses])
        st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.stop()


st.title("A股场内 ETF 短线机会评分台")
st.caption("场内 ETF 页只显示交易所基金体系：实时行情、K线量价、主力资金、盘口、行业热度、成分股实时穿透和场内自选池。")

title_left, title_mid, title_right = st.columns([2.2, 1.1, 1.2])
with title_left:
    st.subheader(f"{code}  {etf_name}")
    st.markdown(f"<span class='small-note'>跟踪标的：{track_target or '-'} ｜ 最新K线：{pd.to_datetime(latest['date']).date()}</span>", unsafe_allow_html=True)
with title_mid:
    st.metric("短线机会评分", f"{model['total_score']:.1f}", help="趋势25%、量能20%、资金20%、板块20%、风险15%")
with title_right:
    st.markdown("状态")
    st.markdown(score_badge(model["action"], model["action_tone"]), unsafe_allow_html=True)
    if st.button("保存评分快照", use_container_width=True):
        try:
            saved_score = save_score_snapshot(code, etf_name, model)
            st.success(f"已保存 {saved_score['rows']} 条")
        except Exception as exc:  # noqa: BLE001
            st.error(f"保存失败：{type(exc).__name__}: {exc}")

latest_price = to_num(spot_row.get("最新价")) if spot_row is not None else latest.get("close")
latest_pct = to_num(spot_row.get("涨跌幅")) if spot_row is not None else latest.get("pct")
latest_amount = to_num(spot_row.get("成交额")) if spot_row is not None else latest.get("amount")
main_amount = raw.get("main_amount")
main_ratio = raw.get("main_ratio")
premium = raw.get("premium")
metric_row1 = st.columns(3)
metric_row2 = st.columns(3)
metric_row1[0].metric("最新价", f"{latest_price:.3f}" if pd.notna(latest_price) else "-")
metric_row1[1].metric("涨跌幅", pct_text(latest_pct))
metric_row1[2].metric("成交额", format_money(latest_amount))
metric_row2[0].metric("主力净额", format_money(main_amount), delta=pct_text(main_ratio))
metric_row2[1].metric("量能倍数", f"{raw.get('amount_ratio5', np.nan):.2f}x", delta=f"20日 {raw.get('amount_ratio20', np.nan):.2f}x")
metric_row2[2].metric("折溢价", pct_text(premium))

st.subheader("场内自选池实时盯盘")
if watchlist_table.empty:
    st.info("场内自选池为空。可以在左侧搜索 ETF 后加入自选。")
else:
    watch_cols = st.columns(5)
    watch_cols[0].metric("自选数量", f"{len(watchlist_table)}")
    if "实时机会分" in watchlist_table:
        watch_cols[1].metric("平均机会分", f"{watchlist_table['实时机会分'].dropna().mean():.1f}" if watchlist_table["实时机会分"].notna().any() else "-")
    if "涨跌幅" in watchlist_table:
        watch_cols[2].metric("上涨数量", f"{int((watchlist_table['涨跌幅'] > 0).sum())}")
    if "主力净流入-净额" in watchlist_table:
        watch_cols[3].metric("主力净额合计", format_money(watchlist_table["主力净流入-净额"].sum()))
    if "盯盘提示" in watchlist_table:
        watch_cols[4].metric("强势观察", f"{int((watchlist_table['盯盘提示'] == '强势观察').sum())}")

    preview = watchlist_table.copy()
    if "实时机会分" in preview:
        preview = preview.sort_values("实时机会分", ascending=False, na_position="last")
    st.dataframe(format_realtime_display(preview.head(12)), use_container_width=True, hide_index=True)
    st.caption("完整自选池和全 ETF 查询仍在“自选池/查询”看板中。")

tabs = st.tabs(["机会评分", "K线与量价", "资金与盘口", "板块与市场", "成分股穿透", "自选池/查询", "模型"])

with tabs[0]:
    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(factor_bar(factor_scores), use_container_width=True)
        st.markdown("**正向证据**")
        st.write("；".join(model["positives"]) if model["positives"] else "暂未形成明显正向合力。")
        st.markdown("**风险证据**")
        st.write("；".join(model["negatives"]) if model["negatives"] else "暂无明显高优先级风险信号。")
    with right:
        st.plotly_chart(radar_chart(factor_scores), use_container_width=True)
    st.divider()
    rule_cols = st.columns(5)
    rule_cols[0].metric("趋势", f"{factor_scores['趋势分']:.1f}")
    rule_cols[1].metric("量能", f"{factor_scores['量能分']:.1f}")
    rule_cols[2].metric("资金", f"{factor_scores['资金分']:.1f}")
    rule_cols[3].metric("板块", f"{factor_scores['板块热度分']:.1f}")
    rule_cols[4].metric("风险", f"{factor_scores['风险控制分']:.1f}")

with tabs[1]:
    st.plotly_chart(candlestick_chart(daily_df, f"{code} {etf_name}：K线 / 均线 / 成交额 / MACD"), use_container_width=True)
    tech_cols = st.columns(7)
    tech_cols[0].metric("MA5", f"{latest.get('ma5', np.nan):.3f}")
    tech_cols[1].metric("MA10", f"{latest.get('ma10', np.nan):.3f}")
    tech_cols[2].metric("MA20", f"{latest.get('ma20', np.nan):.3f}")
    tech_cols[3].metric("MA60", f"{latest.get('ma60', np.nan):.3f}")
    tech_cols[4].metric("RSI14", f"{latest.get('rsi14', np.nan):.1f}")
    tech_cols[5].metric("MFI14", f"{latest.get('mfi14', np.nan):.1f}")
    tech_cols[6].metric("ATR%", pct_text(raw.get("atr_pct")))

with tabs[2]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("实时资金与盘口")
        spot_fields = {
            "IOPV实时估值": to_num(spot_row.get("IOPV实时估值")) if spot_row is not None else np.nan,
            "买一": to_num(spot_row.get("买一")) if spot_row is not None else np.nan,
            "卖一": to_num(spot_row.get("卖一")) if spot_row is not None else np.nan,
            "委比": to_num(spot_row.get("委比")) if spot_row is not None else np.nan,
            "外盘": to_num(spot_row.get("外盘")) if spot_row is not None else np.nan,
            "内盘": to_num(spot_row.get("内盘")) if spot_row is not None else np.nan,
            "主力净流入-净占比": main_ratio,
            "超大单净流入-净占比": to_num(spot_row.get("超大单净流入-净占比")) if spot_row is not None else np.nan,
            "大单净流入-净占比": to_num(spot_row.get("大单净流入-净占比")) if spot_row is not None else np.nan,
        }
        spot_table = pd.DataFrame({"指标": spot_fields.keys(), "数值": spot_fields.values()})
        st.dataframe(spot_table, use_container_width=True, hide_index=True)
    with right:
        st.subheader("实时 ETF 机会榜")
        display_leaderboard = leaderboard.copy()
        for col in ["成交额", "主力净流入-净额"]:
            if col in display_leaderboard:
                display_leaderboard[col] = display_leaderboard[col].map(format_money)
        for col in ["涨跌幅", "基金折价率", "主力净流入-净占比", "实时机会分"]:
            if col in display_leaderboard:
                display_leaderboard[col] = display_leaderboard[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(display_leaderboard, use_container_width=True, hide_index=True)

with tabs[3]:
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("行业热度")
        st.plotly_chart(industry_heat_chart(sector_df), use_container_width=True)
    with right:
        st.subheader("市场环境")
        st.metric("环境分", f"{market_env['score']:.1f}", delta=market_env["regime"])
        st.write(market_env["index_label"])
        st.write(market_env["breadth_label"])
        st.plotly_chart(index_chart(index_df, index_label), use_container_width=True)
        st.subheader("匹配板块")
        sector_info = model["sector_info"]
        if sector_name:
            st.metric("板块", sector_name, delta=pct_text(to_num(sector_info.get("pct_chg"))))
            st.write(f"成交额排名：{to_num(sector_info.get('rank_amount')):.0f} ｜ 资金排名：{to_num(sector_info.get('rank_flow')):.0f} ｜ 上涨占比：{pct_text(to_num(sector_info.get('up_ratio')))}")
            st.write(f"领涨股：{sector_info.get('leader', '-')}")
        else:
            st.write(sector_info.get("note", "未匹配到具体行业。"))

with tabs[4]:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("基金概况")
        overview_display = pd.DataFrame({"项目": list(overview.keys()), "值": list(overview.values())})
        focus = overview_display[overview_display["项目"].isin(["基金全称", "基金类型", "基金管理人", "净资产规模", "份额规模", "管理费率", "托管费率", "跟踪标的"])]
        st.dataframe(focus if not focus.empty else overview_display, use_container_width=True, hide_index=True)
    with right:
        st.subheader("成分股实时穿透")
        impact_cols = st.columns(4)
        impact_cols[0].metric("估算贡献", pct_text(etf_estimated_contribution))
        if etf_holding_impact is not None and not etf_holding_impact.empty and "占净值比例" in etf_holding_impact:
            impact_cols[1].metric("前20权重", pct_text(etf_holding_impact["占净值比例"].dropna().sum()))
        if etf_holding_impact is not None and not etf_holding_impact.empty and "涨跌幅" in etf_holding_impact:
            impact_cols[2].metric("上涨数量", f"{int((etf_holding_impact['涨跌幅'] > 0).sum())}")
        if etf_holding_impact is not None and not etf_holding_impact.empty and "主力净流入-净额" in etf_holding_impact:
            impact_cols[3].metric("主力净额", format_money(numeric_series(etf_holding_impact["主力净流入-净额"]).sum()))

        if etf_holding_impact is not None and not etf_holding_impact.empty:
            latest_quarter = etf_holding_impact["季度"].iloc[0] if "季度" in etf_holding_impact.columns else ""
            if "占净值比例" in etf_holding_impact:
                fig = px.bar(
                    etf_holding_impact.head(20).sort_values("占净值比例"),
                    x="占净值比例",
                    y="股票名称",
                    orientation="h",
                    color="涨跌幅" if "涨跌幅" in etf_holding_impact else None,
                    color_continuous_scale=["#16845b", "#f3d17c", "#cf3f35"],
                    labels={"占净值比例": "占净值比例%", "股票名称": "", "涨跌幅": "实时涨跌%"},
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            display_impact = etf_holding_impact.copy()
            for col in ["占净值比例", "涨跌幅", "估算贡献"]:
                if col in display_impact:
                    display_impact[col] = display_impact[col].map(lambda x: pct_text(x))
            for col in ["成交额", "主力净流入-净额"]:
                if col in display_impact:
                    display_impact[col] = display_impact[col].map(format_money)
            keep_cols = ["股票代码", "股票名称", "占净值比例", "最新价", "涨跌幅", "估算贡献", "成交额", "主力净流入-净额", "季度"]
            st.caption(f"公开持仓期：{latest_quarter}；实时涨跌来自 A 股行情，估算贡献=持仓权重×个股实时涨跌幅。")
            st.dataframe(display_impact[[col for col in keep_cols if col in display_impact.columns]], use_container_width=True, hide_index=True)
        else:
            st.write("未取得持仓数据。")

with tabs[5]:
    st.subheader("基金自选池实时盯盘")
    if watchlist_table.empty:
        st.info("自选池为空。可以在左侧搜索 ETF 后点击“加入自选”，也可以批量编辑代码。")
    else:
        watch_cols = st.columns(4)
        watch_cols[0].metric("自选数量", f"{len(watchlist_table)}")
        if "实时机会分" in watchlist_table:
            watch_cols[1].metric("平均机会分", f"{watchlist_table['实时机会分'].dropna().mean():.1f}" if watchlist_table["实时机会分"].notna().any() else "-")
        if "涨跌幅" in watchlist_table:
            watch_cols[2].metric("上涨数量", f"{int((watchlist_table['涨跌幅'] > 0).sum())}")
        if "主力净流入-净额" in watchlist_table:
            watch_cols[3].metric("主力净流入合计", format_money(watchlist_table["主力净流入-净额"].sum()))

        if "实时机会分" in watchlist_table and "名称" in watchlist_table:
            fig = px.bar(
                watchlist_table.sort_values("实时机会分"),
                x="实时机会分",
                y="名称",
                orientation="h",
                color="涨跌幅" if "涨跌幅" in watchlist_table else None,
                color_continuous_scale=["#16845b", "#f3d17c", "#cf3f35"],
                range_x=[0, 100],
                labels={"实时机会分": "实时机会分", "名称": ""},
            )
            fig.update_layout(height=max(260, min(560, 44 * len(watchlist_table) + 120)), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(format_realtime_display(watchlist_table), use_container_width=True, hide_index=True)
        st.caption("自选池使用实时快照轻量盯盘；点击左侧“分析选中自选”后，主分析区会对该 ETF 运行完整 K 线和五维评分。")

    st.divider()
    st.subheader("全 ETF 实时查询")
    browse_query = st.text_input("表内搜索", value="", placeholder="输入代码、名称或主题关键词", key="browse_etf_query")
    browse_table = build_realtime_etf_table(etf_spot, query=browse_query, limit=300)
    if browse_table.empty:
        st.warning("当前没有匹配的 ETF，或实时行情接口暂不可用。")
    else:
        st.dataframe(format_realtime_display(browse_table), use_container_width=True, hide_index=True)
        st.caption(f"当前展示 {len(browse_table)} 条结果。空搜索默认按实时机会分和成交额排序。")

with tabs[6]:
    st.subheader("场内 ETF 评分公式")
    st.code("ETF短线强度 = 趋势分*0.25 + 量能分*0.20 + 资金分*0.20 + 板块热度分*0.20 + 风险控制分*0.15", language="text")
    st.write(
        "趋势分使用均线多头、突破位置和相对指数强弱；量能分使用今日成交额相对5日/20日均额和量价匹配；"
        "资金分使用主力净流入占比、ETF资金排名、MFI/OBV和近5日吸筹代理；板块热度分使用行业涨幅、成交额、净流入和上涨家数；"
        "风险控制分使用前高空间、20日回撤、折溢价、ATR和短期过热惩罚。"
    )
    st.subheader("动作指示逻辑")
    st.write("禁止追高：近5日涨幅超过10%，5日量能倍数超过2.0，同时折溢价绝对值超过0.45%，且风险控制分低于55。")
    st.write("离场：价格跌破20日线，并且板块热度分低于48或资金分低于45。")
    st.write("减仓：高位放量滞涨，或价格跌破10日线且主力净流出。")
    st.write("买入观察：总分不低于72，趋势、量能、资金和风险均达到最低确认，且市场环境分不低于42。")
    st.write("持有：总分不低于58，未跌破20日线，且未出现明显主力流出。")
    render_model_references()

    st.subheader("数据源状态")
    statuses = [spot_status, *daily_statuses, *index_statuses, overview_status, nav_status, *holdings_statuses, sector_status, a_spot_status]
    status_df = pd.DataFrame([{"数据源": s.get("label"), "状态": "OK" if s.get("ok") else "FAIL", "说明": s.get("detail")} for s in statuses])
    st.dataframe(status_df, use_container_width=True, hide_index=True)
    st.subheader("数据库摘要")
    try:
        summary = database_summary()
        st.write(f"{summary['backend']} ｜ {summary['url']}")
        db_rows = [
            {"表": table, "行数": meta.get("rows"), "最新快照": meta.get("latest")}
            for table, meta in summary.items()
            if isinstance(meta, dict)
        ]
        st.dataframe(pd.DataFrame(db_rows), use_container_width=True, hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"数据库摘要读取失败：{type(exc).__name__}: {exc}")
    st.caption("公网免费接口存在延迟、限流、字段变化和临时不可用。实盘前建议接入券商或交易所授权行情源，并单独做回测和风控校验。")

st.stop()
