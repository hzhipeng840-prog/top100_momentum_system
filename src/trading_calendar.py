from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.paths import A_SHARE_HOLIDAYS_CSV
from src.utils import read_csv_safely


MARKET_TZ = ZoneInfo("Asia/Shanghai")
WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
CAPTURE_READY_TIMES = {
    "intraday_0935": (9, 35),
    "intraday_1030": (10, 30),
    "intraday_1430": (14, 30),
    "intraday_0950": (9, 50),
    "post_close": (16, 0),
}
CAPTURE_WINDOW_GRACE_MINUTES = {
    "intraday_0935": 15,
    "intraday_1030": 20,
    "intraday_1430": 25,
    "intraday_0950": 15,
    "post_close": 90,
}
CAPTURE_TYPE_NAMES = {
    "intraday_0935": "早盘 9:35",
    "intraday_0950": "早盘 9:50",
    "intraday_1030": "早盘 10:30",
    "intraday_1430": "尾盘 14:30",
    "post_close": "收盘后",
}


def load_a_share_holidays() -> dict[str, str]:
    df = read_csv_safely(A_SHARE_HOLIDAYS_CSV)
    if df.empty or "date" not in df.columns:
        return {}
    if "reason" not in df.columns:
        df["reason"] = "交易所休市"
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    return dict(zip(df["date"].astype(str), df["reason"].fillna("交易所休市").astype(str)))


def current_market_time(now: datetime | None = None) -> datetime:
    current = now or datetime.now()
    if current.tzinfo is None:
        return current.astimezone(MARKET_TZ)
    return current.astimezone(MARKET_TZ)


def holiday_reason(day: pd.Timestamp, holidays: dict[str, str] | None = None) -> str | None:
    holidays = holidays if holidays is not None else load_a_share_holidays()
    return holidays.get(pd.Timestamp(day).strftime("%Y-%m-%d"))


def is_a_share_trading_day(day: pd.Timestamp, holidays: dict[str, str] | None = None) -> bool:
    date = pd.Timestamp(day).normalize()
    if date.weekday() >= 5:
        return False
    return holiday_reason(date, holidays=holidays) is None


def previous_a_share_trading_day(day: pd.Timestamp, holidays: dict[str, str] | None = None) -> pd.Timestamp:
    holidays = holidays if holidays is not None else load_a_share_holidays()
    current = pd.Timestamp(day).normalize() - pd.Timedelta(days=1)
    for _ in range(370):
        if is_a_share_trading_day(current, holidays=holidays):
            return current
        current -= pd.Timedelta(days=1)
    raise RuntimeError("无法在本地交易日历中找到上一交易日，请检查 data/calendar/a_share_holidays.csv。")


def latest_expected_market_date(now: datetime | None = None) -> pd.Timestamp:
    current_time = current_market_time(now)
    current = pd.Timestamp(current_time).normalize()
    holidays = load_a_share_holidays()
    if not is_a_share_trading_day(current, holidays=holidays):
        return previous_a_share_trading_day(current, holidays=holidays)
    if current_time.hour < CAPTURE_READY_TIMES["post_close"][0]:
        return previous_a_share_trading_day(current, holidays=holidays)
    return current


def default_signal_date(capture_type: str, now: datetime | None = None) -> str:
    current_time = current_market_time(now)
    current = pd.Timestamp(current_time).normalize()
    holidays = load_a_share_holidays()
    if not is_a_share_trading_day(current, holidays=holidays):
        return previous_a_share_trading_day(current, holidays=holidays).strftime("%Y-%m-%d")
    ready_hour, ready_minute = CAPTURE_READY_TIMES.get(capture_type, CAPTURE_READY_TIMES["post_close"])
    current_minutes = current_time.hour * 60 + current_time.minute
    ready_minutes = ready_hour * 60 + ready_minute
    if current_minutes < ready_minutes:
        return previous_a_share_trading_day(current, holidays=holidays).strftime("%Y-%m-%d")
    return current.strftime("%Y-%m-%d")


def should_skip_market_fetch(capture_type: str, now: datetime | None = None) -> dict:
    current_time = current_market_time(now)
    current = pd.Timestamp(current_time).normalize()
    holidays = load_a_share_holidays()
    expected_signal_date = default_signal_date(capture_type, now=current_time)
    weekday_name = WEEKDAY_NAMES[current.weekday()]

    reason = holiday_reason(current, holidays=holidays)
    if reason is not None:
        return {
            "skip": True,
            "skip_reason_code": "holiday",
            "reason": f"今天是A股休市日（{reason}）。本次跳过新榜抓取，只重算当前缓存。",
            "expected_signal_date": expected_signal_date,
        }
    if current.weekday() >= 5:
        return {
            "skip": True,
            "skip_reason_code": "weekend",
            "reason": f"今天是{weekday_name}，A股不开盘。本次跳过新榜抓取，只重算当前缓存。",
            "expected_signal_date": expected_signal_date,
        }

    ready_hour, ready_minute = CAPTURE_READY_TIMES.get(capture_type, CAPTURE_READY_TIMES["post_close"])
    ready_minutes = ready_hour * 60 + ready_minute
    current_minutes = current_time.hour * 60 + current_time.minute
    if current_minutes < ready_minutes:
        capture_name = CAPTURE_TYPE_NAMES.get(capture_type, capture_type)
        return {
            "skip": True,
            "skip_reason_code": "before_capture",
            "reason": f"当前还没到{capture_name}采集时间。本次跳过新榜抓取，只重算当前缓存。",
            "expected_signal_date": expected_signal_date,
        }

    return {
        "skip": False,
        "skip_reason_code": "",
        "reason": "",
        "expected_signal_date": expected_signal_date,
    }


def is_within_capture_window(
    capture_type: str,
    now: datetime | None = None,
    grace_minutes: int | None = None,
) -> bool:
    current_time = current_market_time(now)
    ready_hour, ready_minute = CAPTURE_READY_TIMES.get(capture_type, CAPTURE_READY_TIMES["post_close"])
    ready_minutes = ready_hour * 60 + ready_minute
    current_minutes = current_time.hour * 60 + current_time.minute
    allowed_grace = int(grace_minutes) if grace_minutes is not None else int(CAPTURE_WINDOW_GRACE_MINUTES.get(capture_type, 20))
    return ready_minutes <= current_minutes <= ready_minutes + allowed_grace
