from __future__ import annotations

import pandas as pd

from src.paths import RAW_STOCK_PRICE_DIR
from src.utils import normalize_code, read_csv_safely


def load_price_data(code: object) -> pd.DataFrame:
    normalized_code = normalize_code(code)
    if not normalized_code:
        return pd.DataFrame()

    path = RAW_STOCK_PRICE_DIR / f"{normalized_code}.csv"
    df = read_csv_safely(path)
    if df.empty:
        return pd.DataFrame()

    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]
    required = {"date", "open", "close", "high", "low"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for column in ["open", "close", "high", "low", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "open", "close", "high", "low"])
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

