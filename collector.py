from __future__ import annotations

import akshare as ak

from db_store import save_etf_spot_snapshot, save_sector_heat_snapshot


def collect_market_snapshot() -> dict[str, object]:
    etf_df = ak.fund_etf_spot_em()
    etf_result = save_etf_spot_snapshot(etf_df)

    sector_result = None
    try:
        sector_df = ak.stock_board_industry_summary_ths()
        sector_result = save_sector_heat_snapshot(sector_df, snapshot_ts=etf_result["snapshot_ts"])
    except Exception as exc:  # noqa: BLE001 - keep ETF snapshot even if sector source is unstable.
        sector_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"etf": etf_result, "sector": sector_result}


if __name__ == "__main__":
    print(collect_market_snapshot())
