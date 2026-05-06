from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"

DATA_ROOT = PROJECT_ROOT / "data"
CALENDAR_DIR = DATA_ROOT / "calendar"
A_SHARE_HOLIDAYS_CSV = CALENDAR_DIR / "a_share_holidays.csv"
RAW_DIR = DATA_ROOT / "raw"
RAW_STOCK_PRICE_DIR = RAW_DIR / "stock_prices"
RAW_INDEX_PRICE_DIR = RAW_DIR / "index_prices"
RAW_POPULARITY_CSV = RAW_DIR / "popularity_top100.csv"
INTRADAY_SNAPSHOT_CSV = RAW_DIR / "intraday_snapshots.csv"
INTRADAY_BAR_DIR = RAW_DIR / "intraday_bars"

PROCESSED_DIR = DATA_ROOT / "processed"
FEATURES_CSV = PROCESSED_DIR / "daily_features.csv"
SIGNALS_CSV = PROCESSED_DIR / "signals.csv"
FOLLOWUPS_CSV = PROCESSED_DIR / "followups.csv"
FAST_STRATEGY_HISTORY_CSV = PROCESSED_DIR / "fast_strategy_history.csv"
MARKET_REGIME_CSV = PROCESSED_DIR / "market_regime.csv"

REPORT_DIR = DATA_ROOT / "reports"
LATEST_PUSH_CSV = REPORT_DIR / "latest_push.csv"
FAST_STRATEGY_CSV = REPORT_DIR / "fast_strategy.csv"
FAST_STRATEGY_AUDIT_CSV = REPORT_DIR / "fast_strategy_audit.csv"
STRONG_RECAP_CSV = REPORT_DIR / "strong_recap.csv"
RULE_EVALUATION_CSV = REPORT_DIR / "rule_evaluation.csv"
LESSON_EVALUATION_CSV = REPORT_DIR / "lesson_evaluation.csv"
BACKTEST_SUMMARY_CSV = REPORT_DIR / "backtest_summary.csv"


DIRECTORIES = [
    CALENDAR_DIR,
    RAW_DIR,
    RAW_STOCK_PRICE_DIR,
    RAW_INDEX_PRICE_DIR,
    INTRADAY_BAR_DIR,
    PROCESSED_DIR,
    REPORT_DIR,
]


def ensure_layout() -> None:
    for folder in DIRECTORIES:
        folder.mkdir(parents=True, exist_ok=True)


def _normalize_version(version: object) -> str:
    text = str(version or "v1").strip().lower()
    return text or "v1"


def versioned_path(path: Path, strategy_version: object = "v1") -> Path:
    normalized = _normalize_version(strategy_version)
    if normalized == "v1":
        return path
    return path.with_name(f"{path.stem}_{normalized}{path.suffix}")


def signals_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(SIGNALS_CSV, strategy_version)


def followups_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(FOLLOWUPS_CSV, strategy_version)


def fast_strategy_history_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(FAST_STRATEGY_HISTORY_CSV, strategy_version)


def latest_push_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(LATEST_PUSH_CSV, strategy_version)


def fast_strategy_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(FAST_STRATEGY_CSV, strategy_version)


def fast_strategy_audit_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(FAST_STRATEGY_AUDIT_CSV, strategy_version)


def strong_recap_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(STRONG_RECAP_CSV, strategy_version)


def rule_evaluation_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(RULE_EVALUATION_CSV, strategy_version)


def lesson_evaluation_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(LESSON_EVALUATION_CSV, strategy_version)


def backtest_summary_csv_for(strategy_version: object = "v1") -> Path:
    return versioned_path(BACKTEST_SUMMARY_CSV, strategy_version)
