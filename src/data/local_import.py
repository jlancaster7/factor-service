"""Import symbol/date source data without warehouse dimension keys."""

import numpy as np
import pandas as pd


PRICE_COLUMNS = {
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "adj_close",
    "volume",
    "change_amount",
    "change_percent",
}
METRIC_COLUMNS = {
    "close_price",
    "market_cap",
    "enterprise_value",
    "pe_ratio",
    "pe_ratio_ttm",
    "pb_ratio",
    "ps_ratio",
    "ps_ratio_ttm",
    "ev_to_revenue",
    "ev_to_revenue_ttm",
    "dividend_yield",
}


def import_frame(db, frame, kind):
    """Atomically upsert prices or point-in-time market metrics.

    Input columns use lower-case names. The caller must supply adjusted prices
    explicitly; raw close prices are never silently treated as adjusted prices.
    """
    if kind not in ("prices", "market-metrics"):
        raise ValueError("kind must be prices or market-metrics")
    frame = frame.copy()
    frame.columns = frame.columns.str.lower()
    value_column = "adj_close" if kind == "prices" else "pb_ratio"
    required = {"symbol", "date", value_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    allowed = PRICE_COLUMNS if kind == "prices" else METRIC_COLUMNS
    unknown = (
        set(frame.columns) - allowed - {"symbol", "date", "sector", "company_name"}
    )
    if unknown:
        raise ValueError(f"Unknown columns: {sorted(unknown)}")
    if frame.empty:
        raise ValueError("No rows to import")
    if frame["symbol"].isna().any() or frame["date"].isna().any():
        raise ValueError("symbol and date must not be null")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if (frame["symbol"] == "").any():
        raise ValueError("symbol must not be empty")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate symbol/date rows in import")
    for col in allowed & set(frame.columns):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col].dropna()).any():
            raise ValueError(f"{col} must not contain infinity")
    if kind == "prices" and (
        frame["adj_close"].isna().any() or (frame["adj_close"] <= 0).any()
    ):
        raise ValueError("adj_close must contain positive, non-null adjusted prices")

    db.initialize_schema()
    db.execute("BEGIN")
    try:
        company_columns = ["symbol"] + [
            c for c in ("sector", "company_name") if c in frame
        ]
        companies = frame[company_columns].drop_duplicates("symbol", keep="last")
        db.merge("ANALYTICS.DIM_COMPANY", _records(companies), ["symbol"])
        dates = pd.DataFrame({"date": frame["date"].drop_duplicates()})
        dates["date_key"] = dates["date"].map(lambda d: int(d.strftime("%Y%m%d")))
        db.merge("ANALYTICS.DIM_DATE", _records(dates), ["date_key"])
        keys = {
            row["SYMBOL"]: row["COMPANY_KEY"]
            for row in db.fetch_all(
                "SELECT symbol, company_key FROM ANALYTICS.DIM_COMPANY"
            )
        }
        values = frame[[c for c in frame.columns if c in allowed]].copy()
        values["company_key"] = frame["symbol"].map(keys)
        values["date_key"] = frame["date"].map(lambda d: int(d.strftime("%Y%m%d")))
        table = "FACT_DAILY_PRICES" if kind == "prices" else "FACT_MARKET_METRICS"
        count = db.merge(
            "ANALYTICS." + table, _records(values), ["company_key", "date_key"]
        )
        db.execute("COMMIT")
        return count
    except Exception:
        db.execute("ROLLBACK")
        raise


def _records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")
