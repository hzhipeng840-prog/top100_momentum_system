from __future__ import annotations

import html
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - older Python builds
    ZoneInfo = None

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtest_queries import (
    BACKTEST_GROUP_NAMES,
    BACKTEST_METRIC_KEY_TO_LABEL,
    build_backtest_compare_table,
    build_backtest_metric_matrix,
    build_backtest_metric_snapshot,
    normalize_backtest_summary,
)
from src.dashboard_metrics import RETURN_METRIC_SPECS, summarize_push_level_performance, summarize_push_level_trend
from src.freshness import build_data_freshness_report
from src.intraday_fetcher import fetch_intraday_bars, fetch_intraday_snapshot
from src.paths import FAST_STRATEGY_AUDIT_CSV, FAST_STRATEGY_CSV, FOLLOWUPS_CSV, LATEST_PUSH_CSV, LESSON_EVALUATION_CSV, MARKET_REGIME_CSV, PROJECT_ROOT, RAW_POPULARITY_CSV, RAW_STOCK_PRICE_DIR, RULE_EVALUATION_CSV, SIGNALS_CSV, STRONG_RECAP_CSV, backtest_summary_csv_for, fast_strategy_audit_csv_for, fast_strategy_csv_for, followups_csv_for, latest_push_csv_for, lesson_evaluation_csv_for, rule_evaluation_csv_for, signals_csv_for, strong_recap_csv_for
from src.pipeline import run_pipeline
from src.settings import load_settings
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, available_strategy_versions, get_strategy_profile, normalize_strategy_version, strategy_default_metric_label
from src.trading_calendar import CAPTURE_TYPE_NAMES as CALENDAR_CAPTURE_TYPE_NAMES, is_a_share_trading_day
from src.utils import normalize_code, read_csv_safely


st.set_page_config(page_title="Top100 情绪动量系统", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 14%, #ffffff 100%);
        }
        .block-container {
            padding-top: 1.55rem;
            padding-bottom: 2rem;
            max-width: 96rem;
        }
        h1 {
            font-size: 2.35rem !important;
        }
        h1, h2, h3 {
            letter-spacing: -0.03em;
        }
        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.4rem 0.8rem;
            font-size: 0.92rem;
        }
        div[data-baseweb="select"] > div {
            min-height: 3rem;
            height: 3rem;
            background: #ffffff !important;
            border: 1px solid rgba(15, 23, 42, 0.16) !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            box-sizing: border-box !important;
            padding: 0 0.58rem 0 0.68rem !important;
        }
        div[data-baseweb="select"] [role="combobox"] {
            min-height: 3rem !important;
            height: 3rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            box-sizing: border-box !important;
        }
        div[data-baseweb="select"] [role="combobox"] > div {
            margin: 0 !important;
            padding: 0 !important;
            text-align: left !important;
        }
        div[data-baseweb="select"] input {
            color: #0f172a !important;
            font-weight: 600 !important;
            -webkit-text-fill-color: #0f172a !important;
            line-height: 1.15 !important;
            text-align: left !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-baseweb="select"] span {
            color: #0f172a !important;
            font-weight: 600 !important;
            line-height: 1.15 !important;
            text-align: left !important;
        }
        div[data-baseweb="select"] svg {
            margin-top: 0 !important;
        }
        div[data-baseweb="popover"] [role="listbox"] {
            background: #ffffff !important;
            border: 1px solid rgba(15, 23, 42, 0.12) !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12) !important;
            padding: 0.35rem !important;
        }
        div[data-baseweb="popover"] [role="option"] {
            color: #0f172a !important;
            font-weight: 550 !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
            background: #e8eef9 !important;
            color: #0f172a !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover {
            background: #f3f6fb !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_SETTINGS = load_settings()
AVAILABLE_STRATEGY_VERSIONS = available_strategy_versions(APP_SETTINGS)
DEFAULT_APP_STRATEGY_VERSION = normalize_strategy_version(APP_SETTINGS.get("default_strategy_version", DEFAULT_STRATEGY_VERSION))
RULE_EVAL_METRIC_LABEL_TO_KEY = {
    "至今收益": "latest",
    "次日开盘收益": "tail_next_open",
    "次日收盘收益": "tail_next_close",
    "1日收益": "1d",
    "3日收益": "3d",
    "5日收益": "5d",
    "10日收益": "10d",
}


DAILY_FLOW_STEPS = [
    ("1", "抓取 Top100", "从本项目接口抓取人气榜，并更新相关个股日 K 缓存。"),
    ("2", "生成特征", "计算排名、连续上榜、涨跌幅、收盘位置、量比、均线偏离。"),
    ("3", "情绪打分", "输出情绪持续分、推送层级、原因和风险。"),
    ("4", "后验跟踪", "结算 1/3/5/10 日收益、最大上涨、最大回撤。"),
    ("5", "更新报表", "刷新快策略、今日推送、样本追踪、强势复盘和规则评估。"),
]

CAPTURE_TYPE_LABELS = {label: capture_type for capture_type, label in CALENDAR_CAPTURE_TYPE_NAMES.items()}
CAPTURE_TYPE_NAMES = {value: label for label, value in CAPTURE_TYPE_LABELS.items()}
SNAPSHOT_STATUS_TYPES = list(CAPTURE_TYPE_LABELS.items())

LESSON_TYPE_NAMES = {
    "pending": "等待后验",
    "pending_data": "行情待补",
    "neutral": "暂未兑现观察",
    "priority_failed": "优先失败复盘",
    "priority_success": "成功特征保留",
    "backup_success": "备选加权参考",
    "low_level_winner": "漏选强势补规则",
}

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo is not None else None


def fmt_pct(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "-"
    return f"{float(numeric):.2f}%"


def render_compact_cards(
    items: list[tuple[str, object]],
    widths: list[float] | None = None,
    label_font_size: str = "0.72rem",
    value_font_size: str = "0.98rem",
    min_height: str = "4.0rem",
) -> None:
    cols = st.columns(widths or [1] * len(items), gap="small")
    for col, (label, value) in zip(cols, items):
        with col:
            safe_label = html.escape(str(label))
            raw_value = str(value)
            safe_value = html.escape(raw_value).replace("\n", "<br>")
            safe_title = html.escape(raw_value).replace('"', "&quot;")
            st.markdown(
                f"""
                <div title="{safe_title}" style="
                    border: 1px solid rgba(15, 23, 42, 0.08);
                    background: rgba(248, 250, 252, 0.84);
                    border-radius: 14px;
                    padding: 0.55rem 0.75rem;
                    min-height: {min_height};
                    box-shadow: 0 1px 1px rgba(15, 23, 42, 0.03);
                ">
                  <div style="font-size: {label_font_size}; color: #64748b; line-height: 1.1; margin-bottom: 0.22rem;">{safe_label}</div>
                  <div style="font-size: {value_font_size}; font-weight: 600; line-height: 1.18; word-break: break-word; overflow-wrap: anywhere;">{safe_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
def current_market_time() -> datetime:
    if SHANGHAI_TZ is not None:
        return datetime.now(SHANGHAI_TZ)
    return datetime.now()


def is_a_share_market_session(now: datetime | None = None) -> bool:
    current = now or current_market_time()
    current_day = pd.Timestamp(current).normalize()
    if not is_a_share_trading_day(current_day):
        return False
    current_minutes = current.hour * 60 + current.minute
    return (9 * 60 + 30) <= current_minutes <= (11 * 60 + 30) or (13 * 60) <= current_minutes <= (15 * 60)


def _snapshot_clock_text(value: object) -> str:
    parsed = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(parsed):
        return pd.Timestamp(parsed).strftime("%H:%M")
    text = str(value or "").strip()
    return text[11:16] if len(text) >= 16 else text


def today_snapshot_status_rows(day: str | None = None) -> list[tuple[str, str, str]]:
    target_day = day or current_market_time().strftime("%Y-%m-%d")
    df = read_csv_safely(RAW_POPULARITY_CSV)
    if df.empty:
        return [(label, "未采", "当前还没有本地 Top100 快照。") for label, _ in SNAPSHOT_STATUS_TYPES]
    required = {"signal_date", "capture_type", "snapshot_time", "code"}
    if not required.issubset(df.columns):
        return [(label, "未知", "本地快照字段不完整。") for label, _ in SNAPSHOT_STATUS_TYPES]

    working = df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["capture_type"] = working["capture_type"].fillna("post_close").astype(str)
    working["snapshot_time"] = working["snapshot_time"].fillna("").astype(str)
    working["code"] = working["code"].apply(normalize_code)
    working = working[working["signal_date"].eq(target_day) & working["code"].ne("")]

    rows: list[tuple[str, str, str]] = []
    for label, capture_type in SNAPSHOT_STATUS_TYPES:
        capture_df = working[working["capture_type"].eq(capture_type)].copy()
        if capture_df.empty:
            rows.append((label, "未采", ""))
            continue

        summary = (
            capture_df.groupby("snapshot_time", dropna=False)["code"]
            .nunique()
            .reset_index(name="code_count")
            .sort_values("snapshot_time")
        )
        complete = summary[summary["code_count"] >= 100].copy()
        if complete.empty:
            latest = summary.iloc[-1]
            rows.append((label, "未满", f"{_snapshot_clock_text(latest['snapshot_time'])}，{int(latest['code_count'])} 只"))
            continue

        times = [_snapshot_clock_text(value) for value in complete["snapshot_time"].tolist()]
        latest = complete.iloc[-1]
        if len(complete) == 1:
            detail = f"{times[-1]}，{int(latest['code_count'])} 只"
        else:
            detail = f"{len(complete)} 次：{', '.join(times[-3:])}"
        rows.append((label, "已采", detail))
    return rows


def render_today_snapshot_status() -> None:
    target_day = current_market_time().strftime("%Y-%m-%d")
    st.sidebar.markdown("### 今日快照状态")
    for label, status, detail in today_snapshot_status_rows(target_day):
        st.sidebar.markdown(f"**{label}**：{status}")
        st.sidebar.caption(detail)


def sync_latest_cloud_results() -> dict[str, object]:
    git_executable = shutil.which("git")
    if not git_executable:
        raise RuntimeError("当前环境没有检测到 git，无法同步云端结果。")

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        [git_executable, "pull", "--ff-only"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    combined_output = "\n".join(part for part in [stdout, stderr] if part).strip()
    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": combined_output,
        "updated": "Already up to date." not in combined_output,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_prev_close(code: str, trade_date: str) -> float | None:
    price_df = read_csv_safely(RAW_STOCK_PRICE_DIR / f"{normalize_code(code)}.csv")
    if price_df.empty or "date" not in price_df.columns or "close" not in price_df.columns:
        return None

    history = price_df.copy()
    history["date"] = history["date"].astype(str)
    history = history[history["date"] < str(trade_date)]
    if history.empty:
        return None

    prev_close = pd.to_numeric(pd.Series([history.iloc[-1].get("close")]), errors="coerce").iloc[0]
    if pd.isna(prev_close):
        return None
    return float(prev_close)


def build_intraday_chart_df(bar_df: pd.DataFrame, prev_close: float | None) -> pd.DataFrame:
    chart_df = bar_df.copy()
    chart_df["datetime"] = pd.to_datetime(chart_df["datetime"], errors="coerce")
    chart_df = chart_df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for column in ["close", "volume", "amount"]:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")

    chart_df["avg_price"] = chart_df["amount"].fillna(0).cumsum() / chart_df["volume"].fillna(0).cumsum().replace({0: float("nan")})
    chart_df["volume_wan"] = chart_df["volume"] / 10000
    chart_df["amount_wan"] = chart_df["amount"] / 10000
    chart_df["amount_yi"] = chart_df["amount"] / 100000000
    chart_df["session_pos"] = chart_df.index.astype(int)
    chart_df["clock"] = chart_df["datetime"].dt.strftime("%H:%M")
    chart_df["datetime_text"] = chart_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if prev_close and prev_close > 0:
        chart_df["price_pct"] = (chart_df["close"] / prev_close - 1) * 100
        chart_df["avg_pct"] = (chart_df["avg_price"] / prev_close - 1) * 100
        chart_df["price_pct_text"] = chart_df["price_pct"].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}%")
        chart_df["avg_pct_text"] = chart_df["avg_pct"].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}%")
    else:
        chart_df["price_pct"] = pd.NA
        chart_df["avg_pct"] = pd.NA
        chart_df["price_pct_text"] = "-"
        chart_df["avg_pct_text"] = "-"
    return chart_df


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    if start == end:
        return [start for _ in range(count)]
    step = (end - start) / (count - 1)
    return [start + step * idx for idx in range(count)]


def _price_axis_range(chart_df: pd.DataFrame, prev_close: float | None) -> tuple[float, float] | None:
    price_values = pd.concat(
        [
            pd.to_numeric(chart_df["close"], errors="coerce"),
            pd.to_numeric(chart_df["avg_price"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if price_values.empty:
        return None

    lower = float(price_values.min())
    upper = float(price_values.max())
    if prev_close and prev_close > 0:
        lower = min(lower, float(prev_close))
        upper = max(upper, float(prev_close))

    span = upper - lower
    pad = max(span * 0.08, max(abs(upper), abs(lower), abs(prev_close or upper)) * 0.004)
    if pad == 0:
        pad = 0.05
    return lower - pad, upper + pad


def _tick_positions(chart_df: pd.DataFrame) -> list[tuple[int, str]]:
    if chart_df.empty:
        return []

    tick_specs = [
        ("09:30", "09:30"),
        ("10:30", "10:30"),
        ("11:30", "11:30/13:00"),
        ("14:00", "14:00"),
        ("15:00", "15:00"),
    ]
    positions: list[tuple[int, str]] = []
    clocks = chart_df["clock"]
    datetimes = chart_df["datetime"]
    trade_date = datetimes.dt.strftime("%Y-%m-%d").iloc[0]

    for target_clock, label in tick_specs:
        exact_matches = chart_df.index[clocks.eq(target_clock)].tolist()
        if exact_matches:
            positions.append((int(exact_matches[0]), label))
            continue

        target_dt = pd.Timestamp(f"{trade_date} {target_clock}:00")
        pos = int(datetimes.searchsorted(target_dt, side="left"))
        pos = min(pos, len(chart_df) - 1)
        positions.append((pos, label))

    deduped: list[tuple[int, str]] = []
    used_positions: set[int] = set()
    for pos, label in positions:
        if pos in used_positions:
            continue
        used_positions.add(pos)
        deduped.append((pos, label))
    return deduped


def build_intraday_price_figure(chart_df: pd.DataFrame, prev_close: float | None) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=chart_df["session_pos"],
            y=chart_df["close"],
            mode="lines",
            name="价格",
            line=dict(color="#2B7DE9", width=2),
            customdata=chart_df[["datetime_text", "price_pct_text"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>价格 %{y:.2f}<br>涨跌 %{customdata[1]}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if chart_df["avg_price"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=chart_df["session_pos"],
                y=chart_df["avg_price"],
                mode="lines",
                name="均价",
                line=dict(color="#F0A202", width=1.8),
                customdata=chart_df[["datetime_text", "avg_pct_text"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>均价 %{y:.2f}<br>涨跌 %{customdata[1]}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if prev_close and prev_close > 0:
        fig.add_hline(
            y=prev_close,
            line_dash="dash",
            line_color="#7A7A7A",
            opacity=0.55,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=chart_df["session_pos"],
                y=chart_df["price_pct"],
                mode="lines",
                line=dict(width=0),
                opacity=0,
                hoverinfo="skip",
                showlegend=False,
                name="涨跌幅",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    price_range = _price_axis_range(chart_df, prev_close)
    if price_range:
        price_min, price_max = price_range
        fig.update_yaxes(range=[price_min, price_max], row=1, col=1, secondary_y=False)
        if prev_close and prev_close > 0:
            pct_min = (price_min / prev_close - 1) * 100
            pct_max = (price_max / prev_close - 1) * 100
            pct_ticks = _linspace(pct_min, pct_max, 5)
            fig.update_yaxes(
                range=[pct_min, pct_max],
                row=1,
                col=1,
                secondary_y=True,
                tickmode="array",
                tickvals=pct_ticks,
                ticktext=[f"{tick:.2f}%" for tick in pct_ticks],
                title_text="涨跌幅",
            )

    if not chart_df.empty:
        ticks = _tick_positions(chart_df)
        tickvals = [pos for pos, _ in ticks]
        ticktext = [label for _, label in ticks]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1, secondary_y=False)
    return fig


def build_intraday_amount_frame(chart_df: pd.DataFrame, step_minutes: int = 5) -> pd.DataFrame:
    if chart_df.empty:
        return chart_df.copy()

    bucket_size = max(int(step_minutes), 1)
    amount_df = chart_df.copy().reset_index(drop=True)
    amount_df["_bucket"] = amount_df.index // bucket_size
    grouped = amount_df.groupby("_bucket", as_index=False).agg(
        session_pos=("session_pos", "first"),
        datetime=("datetime", "last"),
        amount_yi=("amount_yi", "sum"),
    )
    grouped["clock"] = grouped["datetime"].dt.strftime("%H:%M")
    grouped["datetime_text"] = grouped["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return grouped


def build_intraday_amount_figure(chart_df: pd.DataFrame, step_minutes: int = 5) -> go.Figure:
    amount_df = build_intraday_amount_frame(chart_df, step_minutes=step_minutes)
    fig = go.Figure()
    if amount_df.empty:
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
        return fig

    fig.add_trace(
        go.Bar(
            x=amount_df["session_pos"],
            y=amount_df["amount_yi"],
            width=max(step_minutes * 0.75, 0.8),
            name="成交额(亿)",
            marker_color="rgba(43, 125, 233, 0.28)",
            customdata=amount_df[["datetime_text"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>成交额 %{y:.2f}亿<extra></extra>",
        )
    )

    ticks = _tick_positions(amount_df)
    if ticks:
        tickvals = [pos for pos, _ in ticks]
        ticktext = [label for _, label in ticks]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="x unified",
        showlegend=False,
        bargap=0.25,
    )
    fig.update_yaxes(title_text="成交额(亿)")
    return fig


def build_intraday_summary_table(chart_df: pd.DataFrame, step_minutes: int = 5) -> pd.DataFrame:
    if chart_df.empty:
        return chart_df

    summary = chart_df.copy().reset_index(drop=True)
    summary["_pos"] = summary.index
    summary["_clock"] = summary["datetime"].dt.strftime("%H:%M")
    key_times = {"09:30", "09:35", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30", "15:00"}
    sampled = summary[(summary["_pos"] == 0) | (((summary["_pos"] + 1) % step_minutes) == 0) | (summary["_clock"].isin(key_times))].copy()
    sampled = pd.concat([sampled, summary.tail(1)], ignore_index=False)
    sampled = sampled.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    sampled["时间"] = sampled["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sampled["价格"] = sampled["close"].round(2)
    sampled["均价"] = sampled["avg_price"].round(2)
    sampled["涨跌幅"] = sampled["price_pct"].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}%")
    sampled["成交量(万)"] = sampled["volume_wan"].round(1)
    sampled["成交额(万)"] = sampled["amount_wan"].round(1)
    return sampled[["时间", "价格", "均价", "涨跌幅", "成交量(万)", "成交额(万)"]].reset_index(drop=True)


def load_daily_price_row(code: str, trade_date: str) -> pd.DataFrame:
    price_df = read_csv_safely(RAW_STOCK_PRICE_DIR / f"{normalize_code(code)}.csv")
    if price_df.empty or "date" not in price_df.columns:
        return pd.DataFrame()
    matched = price_df[price_df["date"].astype(str).eq(str(trade_date))].copy()
    return matched.tail(1)


def show_daily_price_fallback(code: str, trade_date: str, bar_source: str) -> None:
    daily_row = load_daily_price_row(code, trade_date)
    if daily_row.empty:
        st.info("暂无分钟线数据，也没有找到该日期的本地日线缓存。")
        return

    if str(bar_source).startswith("remote_failed"):
        st.warning("分钟线接口远程断连；已找到本地日线数据，先显示开高低收兜底。")
    elif str(bar_source).startswith("empty_remote"):
        st.warning("分钟线接口没有返回数据；已找到本地日线数据，先显示开高低收兜底。")
    else:
        st.info("暂无分钟线数据；已找到本地日线数据，先显示开高低收兜底。")

    row = daily_row.iloc[-1]
    fallback_cols = st.columns(5)
    fallback_cols[0].metric("日线开盘", row.get("open", "-"))
    fallback_cols[1].metric("日线收盘", row.get("close", "-"))
    fallback_cols[2].metric("日线最高", row.get("high", "-"))
    fallback_cols[3].metric("日线最低", row.get("low", "-"))
    fallback_cols[4].metric("成交量", row.get("volume", "-"))


def display_table(
    df: pd.DataFrame,
    columns: list[str],
    rename: dict[str, str],
    limit: int | None = None,
    column_config: dict[str, Any] | None = None,
) -> None:
    if df.empty:
        st.info("暂无数据。先运行 `python daily_job.py`。")
        return
    available = [column for column in columns if column in df.columns]
    display_df = df[available].copy()
    if limit:
        display_df = display_df.head(limit)
    if "capture_type" in display_df.columns:
        display_df["capture_type"] = display_df["capture_type"].map(CAPTURE_TYPE_NAMES).fillna(display_df["capture_type"])
    if "lesson_type" in display_df.columns:
        display_df["lesson_type"] = display_df["lesson_type"].map(LESSON_TYPE_NAMES).fillna(display_df["lesson_type"])
    display_df = display_df.rename(columns=rename)
    display_df = stringify_display_df(display_df)
    st.dataframe(display_df, width="stretch", hide_index=True, column_config=column_config)


def stringify_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    for column in display_df.columns:
        series = display_df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue

        numeric_series = pd.to_numeric(series, errors="coerce")
        if series.notna().any() and numeric_series[series.notna()].notna().all():
            display_df[column] = numeric_series
            continue

        display_df[column] = series.where(pd.notna(series), "-").astype(str)
    return display_df


def format_table_df(df: pd.DataFrame, columns: list[str], rename: dict[str, str], limit: int | None = None) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    display_df = df[available].copy()
    if limit:
        display_df = display_df.head(limit)
    if "capture_type" in display_df.columns:
        display_df["capture_type"] = display_df["capture_type"].map(CAPTURE_TYPE_NAMES).fillna(display_df["capture_type"])
    if "lesson_type" in display_df.columns:
        display_df["lesson_type"] = display_df["lesson_type"].map(LESSON_TYPE_NAMES).fillna(display_df["lesson_type"])
    display_df = display_df.rename(columns=rename)
    return stringify_display_df(display_df)


PUSH_LEVEL_COLORS = {
    "强推观察": "#D1495B",
    "重点观察": "#EDAE49",
    "普通观察": "#2B7DE9",
    "观察池": "#7C3AED",
    "不推送": "#9AA5B1",
}


def _push_level_color(push_level: str) -> str:
    return PUSH_LEVEL_COLORS.get(str(push_level), "#5C677D")


def build_push_level_bar_figure(summary_df: pd.DataFrame, metric_label: str, signal_date: str) -> go.Figure:
    valid_df = summary_df.copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("上涨胜率", f"{metric_label}均值"),
        horizontal_spacing=0.12,
    )
    if valid_df.empty:
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=54, b=10), title=f"{signal_date} 推送分层表现")
        return fig

    valid_df["valid_count"] = pd.to_numeric(valid_df["valid_count"], errors="coerce").fillna(0)
    valid_df["win_rate_pct"] = pd.to_numeric(valid_df["win_rate_pct"], errors="coerce")
    valid_df["avg_return_pct"] = pd.to_numeric(valid_df["avg_return_pct"], errors="coerce")
    valid_df = valid_df[valid_df["valid_count"] > 0].copy()
    if valid_df.empty:
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=54, b=10),
            title=f"{signal_date} 推送分层表现",
            annotations=[
                dict(
                    text="当前观察窗口还没有可结算样本",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    colors = [_push_level_color(level) for level in valid_df["push_level"]]
    customdata = valid_df[["sample_count", "valid_count", "up_count", "pending_count"]].to_numpy()
    fig.add_trace(
        go.Bar(
            x=valid_df["push_level"],
            y=valid_df["win_rate_pct"],
            marker_color=colors,
            text=valid_df["win_rate_pct"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%"),
            textposition="outside",
            customdata=customdata,
            hovertemplate="层级 %{x}<br>上涨胜率 %{y:.2f}%<br>样本数 %{customdata[0]}<br>已结算 %{customdata[1]}<br>上涨数 %{customdata[2]}<br>待结算 %{customdata[3]}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=valid_df["push_level"],
            y=valid_df["avg_return_pct"],
            marker_color=colors,
            text=valid_df["avg_return_pct"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}%"),
            textposition="outside",
            customdata=customdata,
            hovertemplate="层级 %{x}<br>平均收益 %{y:.2f}%<br>样本数 %{customdata[0]}<br>已结算 %{customdata[1]}<br>上涨数 %{customdata[2]}<br>待结算 %{customdata[3]}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="胜率 %", row=1, col=1)
    fig.update_yaxes(title_text="平均收益 %", row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=54, b=10), title=f"{signal_date} 推送分层表现")
    return fig


def build_push_level_bubble_figure(summary_df: pd.DataFrame, metric_label: str, signal_date: str) -> go.Figure:
    valid_df = summary_df.copy()
    fig = go.Figure()
    if valid_df.empty:
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    valid_df["valid_count"] = pd.to_numeric(valid_df["valid_count"], errors="coerce").fillna(0)
    valid_df["win_rate_pct"] = pd.to_numeric(valid_df["win_rate_pct"], errors="coerce")
    valid_df["avg_return_pct"] = pd.to_numeric(valid_df["avg_return_pct"], errors="coerce")
    valid_df = valid_df[valid_df["valid_count"] > 0].copy()
    if valid_df.empty:
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=54, b=10),
            title=f"{signal_date} 推送分层气泡图",
            annotations=[
                dict(
                    text="当前观察窗口还没有可结算样本",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    marker_sizes = valid_df["valid_count"].clip(lower=1).astype(float) * 10 + 18
    fig.add_trace(
        go.Scatter(
            x=valid_df["avg_return_pct"],
            y=valid_df["win_rate_pct"],
            mode="markers+text",
            text=valid_df["push_level"],
            textposition="top center",
            marker=dict(
                size=marker_sizes,
                color=[_push_level_color(level) for level in valid_df["push_level"]],
                opacity=0.78,
                line=dict(color="#FFFFFF", width=1),
            ),
            customdata=valid_df[["sample_count", "valid_count", "up_count", "pending_count"]].to_numpy(),
            hovertemplate="层级 %{text}<br>平均收益 %{x:.2f}%<br>上涨胜率 %{y:.2f}%<br>样本数 %{customdata[0]}<br>已结算 %{customdata[1]}<br>上涨数 %{customdata[2]}<br>待结算 %{customdata[3]}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=54, b=10),
        title=f"{signal_date} 推送分层气泡图",
        xaxis_title=f"{metric_label}均值 %",
        yaxis_title="上涨胜率 %",
    )
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(15, 23, 42, 0.25)")
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(15, 23, 42, 0.25)")
    return fig


def build_push_level_trend_figure(trend_df: pd.DataFrame, metric_label: str, title: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("上涨胜率趋势", f"{metric_label}均值趋势"),
        horizontal_spacing=0.12,
    )
    if trend_df.empty:
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=54, b=10), title=title)
        return fig

    working = trend_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working["valid_count"] = pd.to_numeric(working["valid_count"], errors="coerce").fillna(0)
    working["sample_count"] = pd.to_numeric(working["sample_count"], errors="coerce").fillna(0)
    working["pending_count"] = pd.to_numeric(working["pending_count"], errors="coerce").fillna(0)
    working["up_count"] = pd.to_numeric(working["up_count"], errors="coerce").fillna(0)
    working["win_rate_pct"] = pd.to_numeric(working["win_rate_pct"], errors="coerce")
    working["avg_return_pct"] = pd.to_numeric(working["avg_return_pct"], errors="coerce")
    working = working.dropna(subset=["signal_date"]).copy()
    working = working[working["valid_count"] > 0].copy()
    if working.empty:
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=54, b=10),
            title=title,
            annotations=[
                dict(
                    text="当前范围内还没有可结算样本",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    ordered_levels = [level for level in PUSH_LEVEL_COLORS if level in set(working["push_level"].astype(str))]
    ordered_levels.extend(sorted(level for level in set(working["push_level"].astype(str)) if level not in ordered_levels))
    for push_level in ordered_levels:
        level_df = working[working["push_level"].astype(str).eq(push_level)].sort_values("signal_date").copy()
        if level_df.empty:
            continue
        color = _push_level_color(push_level)
        customdata = level_df[["sample_count", "valid_count", "up_count", "pending_count"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=level_df["signal_date"],
                y=level_df["win_rate_pct"],
                mode="lines+markers",
                name=push_level,
                legendgroup=push_level,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                customdata=customdata,
                hovertemplate="日期 %{x|%Y-%m-%d}<br>层级 %{fullData.name}<br>上涨胜率 %{y:.2f}%<br>样本数 %{customdata[0]}<br>已结算 %{customdata[1]}<br>上涨数 %{customdata[2]}<br>待结算 %{customdata[3]}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=level_df["signal_date"],
                y=level_df["avg_return_pct"],
                mode="lines+markers",
                name=push_level,
                legendgroup=push_level,
                showlegend=False,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                customdata=customdata,
                hovertemplate="日期 %{x|%Y-%m-%d}<br>层级 %{fullData.name}<br>平均收益 %{y:.2f}%<br>样本数 %{customdata[0]}<br>已结算 %{customdata[1]}<br>上涨数 %{customdata[2]}<br>待结算 %{customdata[3]}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    fig.update_yaxes(title_text="胜率 %", row=1, col=1)
    fig.update_yaxes(title_text="平均收益 %", row=1, col=2)
    fig.update_xaxes(title_text="样本日期", tickformat="%m-%d", row=1, col=1)
    fig.update_xaxes(title_text="样本日期", tickformat="%m-%d", row=1, col=2)
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=54, b=10), title=title, hovermode="x unified")
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(15, 23, 42, 0.18)", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(15, 23, 42, 0.18)", row=1, col=2)
    return fig


def _group_value_color(group_value: str) -> str:
    if str(group_value) in PUSH_LEVEL_COLORS:
        return _push_level_color(str(group_value))
    palette = {
        "Top3": "#D1495B",
        "Top10": "#EDAE49",
        "Top20": "#2B7DE9",
        "Top50": "#4CB944",
        "Top100": "#5C677D",
        "首次上榜": "#2B7DE9",
        "连续第2天": "#EDAE49",
        "连续第3天": "#D1495B",
        "连续4天及以上": "#5C677D",
    }
    return palette.get(str(group_value), "#5C677D")


def build_backtest_group_bar_figure(summary_df: pd.DataFrame, metric_label: str, title: str) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("上涨胜率", f"{metric_label}均值"),
        horizontal_spacing=0.12,
    )
    if summary_df.empty:
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=54, b=10), title=title)
        return fig

    working = summary_df.copy()
    for column in ["sample_count", "pushed_count", "valid_count", "win_rate_pct", "avg_return_pct", "strong_rate_pct"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working[pd.to_numeric(working["valid_count"], errors="coerce").fillna(0) > 0].copy()
    if working.empty:
        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=54, b=10),
            title=title,
            annotations=[
                dict(
                    text="当前筛选范围内还没有可结算样本",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
        )
        return fig

    colors = [_group_value_color(value) for value in working["group_value"]]
    customdata = working[["sample_count", "pushed_count", "valid_count", "strong_rate_pct"]].to_numpy()
    fig.add_trace(
        go.Bar(
            x=working["group_value"],
            y=working["win_rate_pct"],
            marker_color=colors,
            text=working["win_rate_pct"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.1f}%"),
            textposition="outside",
            customdata=customdata,
            hovertemplate="分组 %{x}<br>上涨胜率 %{y:.2f}%<br>样本数 %{customdata[0]}<br>推送数 %{customdata[1]}<br>已结算 %{customdata[2]}<br>强势率 %{customdata[3]:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=working["group_value"],
            y=working["avg_return_pct"],
            marker_color=colors,
            text=working["avg_return_pct"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}%"),
            textposition="outside",
            customdata=customdata,
            hovertemplate="分组 %{x}<br>平均收益 %{y:.2f}%<br>样本数 %{customdata[0]}<br>推送数 %{customdata[1]}<br>已结算 %{customdata[2]}<br>强势率 %{customdata[3]:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="胜率 %", row=1, col=1)
    fig.update_yaxes(title_text="平均收益 %", row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=54, b=10), title=title)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(15, 23, 42, 0.18)", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(15, 23, 42, 0.18)", row=1, col=2)
    return fig


def version_display_name(strategy_version: str) -> str:
    return str(get_strategy_profile(strategy_version).get("name", strategy_version.upper()))


def merge_version_frames(
    frame_by_version: dict[str, pd.DataFrame],
    key_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for strategy_version, frame in frame_by_version.items():
        working = frame.copy()
        for column in key_columns + value_columns:
            if column not in working.columns:
                working[column] = None
        working = working[key_columns + value_columns].copy()
        renamed = working.rename(columns={column: f"{strategy_version}_{column}" for column in value_columns})
        merged = renamed if merged is None else merged.merge(renamed, on=key_columns, how="outer")
    return merged if merged is not None else pd.DataFrame(columns=key_columns)


def build_push_level_compare_table(summary_by_version: dict[str, pd.DataFrame]) -> pd.DataFrame:
    versions = [version for version, frame in summary_by_version.items() if isinstance(frame, pd.DataFrame)]
    if not versions:
        return pd.DataFrame()

    compare_df = merge_version_frames(
        {version: summary_by_version.get(version, pd.DataFrame()) for version in versions},
        key_columns=["push_level"],
        value_columns=[
            "sample_count",
            "valid_count",
            "pending_count",
            "up_count",
            "win_rate_pct",
            "avg_return_pct",
            "median_return_pct",
        ],
    )
    baseline = versions[0]
    for version in versions[1:]:
        left_win = f"{baseline}_win_rate_pct"
        right_win = f"{version}_win_rate_pct"
        if {left_win, right_win}.issubset(compare_df.columns):
            compare_df[f"{version}_vs_{baseline}_win_rate_delta_pct"] = pd.to_numeric(compare_df[right_win], errors="coerce") - pd.to_numeric(compare_df[left_win], errors="coerce")
        left_avg = f"{baseline}_avg_return_pct"
        right_avg = f"{version}_avg_return_pct"
        if {left_avg, right_avg}.issubset(compare_df.columns):
            compare_df[f"{version}_vs_{baseline}_avg_return_delta_pct"] = pd.to_numeric(compare_df[right_avg], errors="coerce") - pd.to_numeric(compare_df[left_avg], errors="coerce")
        left_up = f"{baseline}_up_count"
        right_up = f"{version}_up_count"
        if {left_up, right_up}.issubset(compare_df.columns):
            compare_df[f"{version}_vs_{baseline}_up_count_delta"] = pd.to_numeric(compare_df[right_up], errors="coerce") - pd.to_numeric(compare_df[left_up], errors="coerce")

    push_level_order = {"强推观察": 0, "重点观察": 1, "普通观察": 2, "观察池": 3, "不推送": 4}
    compare_df["_order"] = compare_df["push_level"].map(push_level_order).fillna(99)
    compare_df = compare_df.sort_values(["_order", "push_level"]).drop(columns=["_order"]).reset_index(drop=True)
    return compare_df


def build_rule_eval_compare_table(rule_eval_by_version: dict[str, pd.DataFrame], metric_key: str) -> pd.DataFrame:
    versions = [version for version, frame in rule_eval_by_version.items() if isinstance(frame, pd.DataFrame)]
    if not versions:
        return pd.DataFrame()

    value_columns = [
        "sample_count",
        "pushed_count",
        f"valid_{metric_key}",
        f"avg_{metric_key}",
        f"win_rate_{metric_key}",
        f"strong_rate_{metric_key}",
    ]
    compare_df = merge_version_frames(
        {version: rule_eval_by_version.get(version, pd.DataFrame()) for version in versions},
        key_columns=["group_name", "group_value"],
        value_columns=value_columns,
    )
    baseline = versions[0]
    delta_columns = [
        "pushed_count",
        f"avg_{metric_key}",
        f"win_rate_{metric_key}",
        f"strong_rate_{metric_key}",
    ]
    for version in versions[1:]:
        for column in delta_columns:
            left = f"{baseline}_{column}"
            right = f"{version}_{column}"
            if {left, right}.issubset(compare_df.columns):
                compare_df[f"{version}_vs_{baseline}_{column}_delta"] = pd.to_numeric(compare_df[right], errors="coerce") - pd.to_numeric(compare_df[left], errors="coerce")

    group_order = {"推送层级": 0, "人气排名段": 1, "上榜阶段": 2}
    group_value_order = {
        "强推观察": 0,
        "重点观察": 1,
        "普通观察": 2,
        "观察池": 3,
        "不推送": 4,
        "Top3": 10,
        "Top10": 11,
        "Top20": 12,
        "Top50": 13,
        "Top100": 14,
        "首次上榜": 20,
        "连续第2天": 21,
        "连续第3天": 22,
        "连续4天及以上": 23,
    }
    compare_df["_group_order"] = compare_df["group_name"].map(group_order).fillna(99)
    compare_df["_value_order"] = compare_df["group_value"].map(group_value_order).fillna(99)
    compare_df = compare_df.sort_values(["_group_order", "_value_order", "group_name", "group_value"]).drop(columns=["_group_order", "_value_order"]).reset_index(drop=True)
    return compare_df


def is_priority_focus(row: pd.Series, candidate_hit: str) -> str:
    if candidate_hit != "是":
        return "否"
    return "是" if str(row.get("push_level", "")) in {"强推观察", "重点观察"} else "否"


def build_execution_display_df(push_df: pd.DataFrame, fast_df: pd.DataFrame, strategy_version: str) -> pd.DataFrame:
    if push_df.empty:
        return pd.DataFrame()

    working_df = push_df.reset_index(drop=True).copy()
    if "push_level" in working_df.columns:
        # Execution list should only surface actionable names; observation pool stays out.
        working_df = working_df[working_df["push_level"].astype(str).ne("观察池")].copy()
    if strategy_version in {"v2", "v3"} and "push_level" in working_df.columns:
        # For intraday versions, once limit-up names are removed, execution rows must stay high quality.
        working_df = working_df[working_df["push_level"].astype(str).isin(["强推观察", "重点观察"])].copy()
    if working_df.empty:
        return pd.DataFrame()

    fast_lookup = pd.DataFrame()
    if not fast_df.empty and "code" in fast_df.columns:
        fast_lookup = fast_df.copy()
        fast_lookup["_norm_code"] = fast_lookup["code"].astype(str).map(normalize_code)

    display_df = pd.DataFrame(index=working_df.index)
    display_df["查看"] = False
    display_df["股票"] = working_df["name"].astype(str) + " " + working_df["code"].astype(str).map(normalize_code)
    display_df["推送层级"] = working_df.get("push_level", "-")
    display_df["当日涨跌"] = working_df.get("day_return_pct", "-").apply(fmt_pct)
    display_df["关键理由"] = working_df.get("reasons", "-")
    hit_rows = []

    priority_rows = []
    for _, row in working_df.iterrows():
        enriched = row.copy()
        candidate_hit = "否"
        if not fast_lookup.empty:
            matches = fast_lookup[fast_lookup["_norm_code"].eq(normalize_code(row.get("code", "")))]
            if not matches.empty:
                candidate_hit = "是"
                fast_row = matches.iloc[0]
                for column in ["fast_level", "next_session_plan"]:
                    enriched[column] = fast_row.get(column, enriched.get(column, ""))
        hit_rows.append(candidate_hit)
        priority_rows.append(is_priority_focus(enriched, candidate_hit))
    display_df["候选池命中"] = hit_rows
    display_df["是否优先关注"] = priority_rows
    return display_df


def build_detail_row(push_row: pd.Series, fast_df: pd.DataFrame) -> dict[str, Any]:
    detail = push_row.to_dict()
    if fast_df.empty or "code" not in fast_df.columns:
        return detail

    norm_code = normalize_code(push_row.get("code", ""))
    matches = fast_df[fast_df["code"].astype(str).map(normalize_code).eq(norm_code)]
    if matches.empty:
        return detail

    fast_detail = matches.iloc[0].to_dict()
    for key, value in fast_detail.items():
        if key not in detail or pd.isna(detail.get(key)) or str(detail.get(key, "")).strip() in {"", "-"}:
            detail[key] = value
    for key in ["strategy_date", "training_date", "analysis_window", "learned_rule", "fast_score", "fast_level", "next_session_plan", "fit_reasons"]:
        detail[key] = fast_detail.get(key, detail.get(key, "-"))
    return detail


def render_stock_detail_dialog(row: dict[str, Any], default_trade_date: str) -> None:
    code = str(row.get("code", "-"))
    name = str(row.get("name", "-"))

    @st.dialog(f"{name} {code}", width="large")
    def _dialog() -> None:
        render_compact_cards(
            [
                ("人气排名", row.get("rank", "-")),
                ("快策略层级", row.get("fast_level", "-")),
                ("策略分", row.get("fast_score", "-")),
                ("原始层级", row.get("push_level", "-")),
                ("当日涨跌", fmt_pct(row.get("day_return_pct"))),
            ],
            widths=[0.8, 1.7, 0.8, 1.1, 0.8],
            value_font_size="1.02rem",
            min_height="3.75rem",
        )

        st.markdown(f"**快策略命中：** {row.get('fit_reasons', '-')}")
        st.markdown(f"**原始原因：** {row.get('reasons', '-')}")
        st.markdown(f"**风险：** {row.get('risks', '-')}")
        st.markdown("**下个交易日计划**")
        st.info(str(row.get("next_session_plan", "-")))

        render_compact_cards(
            [
                ("近5日", fmt_pct(row.get("pre5_return_pct"))),
                ("5日量比", row.get("volume_ratio_5", "-")),
                ("收盘位置", row.get("close_position", "-")),
                ("采集类型", CAPTURE_TYPE_NAMES.get(str(row.get("capture_type", "")), row.get("capture_type", "-"))),
            ],
            widths=[1, 1, 1, 1],
            value_font_size="1.00rem",
            min_height="3.65rem",
        )
        st.caption(f"快照时间：{row.get('snapshot_time', '-')}")

        st.divider()
        control_cols = st.columns([0.78, 3.15, 1.0], gap="small", vertical_alignment="center")
        with control_cols[0]:
            st.markdown("<div style='font-size:0.92rem; font-weight:600; line-height:1.2;'>走势日期</div>", unsafe_allow_html=True)
        with control_cols[1]:
            trade_date_input = st.text_input("走势日期", value=default_trade_date, label_visibility="collapsed", key=f"detail_trade_date_{code}")
        with control_cols[2]:
            load_pressed = st.button("查看盘中走势", type="secondary", key=f"detail_load_{code}")

        if load_pressed:
            try:
                trade_date_parsed = pd.to_datetime(trade_date_input, errors="coerce")
                if pd.isna(trade_date_parsed):
                    st.error("走势日期格式不正确，请输入 YYYY-MM-DD。")
                    return
                trade_date = trade_date_parsed.strftime("%Y-%m-%d")
                now_cn = current_market_time()
                live_session = is_a_share_market_session(now_cn)
                selected_is_today = trade_date == now_cn.strftime("%Y-%m-%d")
                with st.spinner("正在按需获取盘中数据..."):
                    snapshot_df, snapshot_source = fetch_intraday_snapshot(
                        code,
                        capture_type="manual",
                        force_refresh=live_session,
                    )
                    bar_df, bar_source = fetch_intraday_bars(
                        code,
                        trade_date=trade_date,
                        period="1",
                        force_refresh=live_session and selected_is_today,
                    )
                st.caption(f"快照来源：{snapshot_source}；分钟线来源：{bar_source}")
                if not snapshot_df.empty:
                    snap = snapshot_df.iloc[-1]
                    render_compact_cards(
                        [
                            ("最新价", snap.get("last_price", "-")),
                            ("当前涨跌", fmt_pct(snap.get("current_return_pct"))),
                            ("盘中最高", snap.get("day_high_so_far", "-")),
                            ("盘中最低", snap.get("day_low_so_far", "-")),
                            ("量比", snap.get("volume_ratio", "-")),
                        ],
                        widths=[1, 1, 1, 1, 1],
                        value_font_size="1.00rem",
                        min_height="3.65rem",
                    )
                if not bar_df.empty:
                    prev_close = None
                    if not snapshot_df.empty and "prev_close" in snapshot_df.columns:
                        prev_close_value = pd.to_numeric(pd.Series([snapshot_df.iloc[-1].get("prev_close")]), errors="coerce").iloc[0]
                        if not pd.isna(prev_close_value):
                            prev_close = float(prev_close_value)
                    if prev_close is None:
                        prev_close = load_prev_close(code, trade_date)

                    chart_df = build_intraday_chart_df(bar_df, prev_close)
                    chart_cols = st.columns([1.6, 1.0], gap="medium")
                    with chart_cols[0]:
                        st.plotly_chart(
                            build_intraday_price_figure(chart_df, prev_close),
                            width="stretch",
                            config={"displayModeBar": False},
                        )
                    with chart_cols[1]:
                        st.plotly_chart(
                            build_intraday_amount_figure(chart_df, step_minutes=5),
                            width="stretch",
                            config={"displayModeBar": False},
                        )

                    with st.expander("查看 5 分钟采样", expanded=False):
                        summary_df = build_intraday_summary_table(chart_df, step_minutes=5)
                        st.dataframe(stringify_display_df(summary_df), width="stretch", hide_index=True)

                    with st.expander("查看完整分钟明细", expanded=False):
                        display_table(
                            bar_df,
                            columns=["datetime", "open", "close", "high", "low", "volume", "amount"],
                            rename={
                                "datetime": "时间",
                                "open": "开",
                                "close": "收",
                                "high": "高",
                                "low": "低",
                                "volume": "成交量",
                                "amount": "成交额",
                            },
                            limit=120,
                        )
                else:
                    show_daily_price_fallback(code, trade_date, bar_source)
                    st.info("暂无分钟线数据，可能是非交易日、接口未返回，或本地还没有缓存。")
            except Exception as exc:
                st.error(f"盘中数据获取失败：{exc}")

    _dialog()

def dataset_paths_for_version(strategy_version: str) -> dict[str, Path]:
    normalized = normalize_strategy_version(strategy_version)
    return {
        "signals": signals_csv_for(normalized),
        "followups": followups_csv_for(normalized),
        "backtest_summary": backtest_summary_csv_for(normalized),
        "latest_push": latest_push_csv_for(normalized),
        "fast_strategy": fast_strategy_csv_for(normalized),
        "fast_strategy_audit": fast_strategy_audit_csv_for(normalized),
        "strong_recap": strong_recap_csv_for(normalized),
        "rule_eval": rule_evaluation_csv_for(normalized),
        "lesson_eval": lesson_evaluation_csv_for(normalized),
        "market_regime": MARKET_REGIME_CSV,
    }


def _file_signature(path: Path) -> tuple[str, int, int]:
    resolved = Path(path)
    if not resolved.exists():
        return str(resolved), 0, 0
    stat = resolved.stat()
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@st.cache_data(show_spinner=False)
def _load_csv_cached(path_str: str, modified_ns: int, size: int) -> pd.DataFrame:
    del modified_ns, size
    return read_csv_safely(Path(path_str))


def _load_dataset(path: Path) -> pd.DataFrame:
    path_str, modified_ns, size = _file_signature(path)
    return _load_csv_cached(path_str, modified_ns, size)


def load_all(strategy_version: str) -> dict[str, pd.DataFrame]:
    dataset_paths = dataset_paths_for_version(strategy_version)
    return {
        key: _load_dataset(path)
        for key, path in dataset_paths.items()
    }


def render_sidebar_daily_flow() -> str:
    st.sidebar.title("工作区")

    strategy_label_map = {version: get_strategy_profile(version).get("name", version.upper()) for version in AVAILABLE_STRATEGY_VERSIONS}
    default_strategy_index = AVAILABLE_STRATEGY_VERSIONS.index(DEFAULT_APP_STRATEGY_VERSION) if DEFAULT_APP_STRATEGY_VERSION in AVAILABLE_STRATEGY_VERSIONS else 0
    selected_strategy_version = st.sidebar.selectbox(
        "策略版本",
        options=AVAILABLE_STRATEGY_VERSIONS,
        index=default_strategy_index,
        format_func=lambda version: strategy_label_map.get(version, version.upper()),
        key="dashboard_strategy_version",
    )
    selected_strategy_profile = get_strategy_profile(selected_strategy_version)
    with st.sidebar.expander("版本说明", expanded=False):
        st.write(str(selected_strategy_profile.get("description", "")))

    render_today_snapshot_status()

    st.sidebar.markdown("### 云端同步")
    if st.sidebar.button("同步最新云端结果", type="secondary", width="stretch"):
        try:
            with st.spinner("正在从 GitHub 同步最新结果..."):
                sync_result = sync_latest_cloud_results()
            st.session_state["last_cloud_sync_result"] = sync_result
            st.session_state.pop("last_cloud_sync_error", None)
            _load_csv_cached.clear()
            if sync_result.get("success"):
                if sync_result.get("updated"):
                    st.sidebar.success("已同步到最新云端结果，当前页面正在刷新。")
                else:
                    st.sidebar.info("云端没有更新提交，本地已经是最新。")
                st.rerun()
            st.sidebar.error(f"同步失败：{sync_result.get('output') or 'git pull 执行失败'}")
        except Exception as exc:
            st.session_state["last_cloud_sync_error"] = str(exc)
            st.sidebar.error(f"同步失败：{exc}")

    if "last_cloud_sync_result" in st.session_state:
        sync_result = st.session_state["last_cloud_sync_result"]
        with st.sidebar.expander("最近一次云端同步", expanded=False):
            st.write(f"完成：{sync_result.get('finished_at', '-')}")
            st.write(f"状态：{'成功' if sync_result.get('success') else '失败'}")
            st.write(f"返回码：{sync_result.get('returncode', '-')}")
            st.code(str(sync_result.get("output") or "无输出"))

    if "last_cloud_sync_error" in st.session_state:
        with st.sidebar.expander("最近一次同步错误", expanded=False):
            st.code(st.session_state["last_cloud_sync_error"])

    st.sidebar.markdown("### 今日操作")
    recompute_only = st.sidebar.checkbox(
        "只重算当前缓存",
        value=False,
        help="不联网抓取，只用当前 data 目录里的数据重算报表。",
    )
    capture_type_label = st.sidebar.selectbox(
        "采集类型",
        options=list(CAPTURE_TYPE_LABELS.keys()),
        index=0,
    )
    snapshot_time = st.sidebar.text_input(
        "快照时间",
        value="",
        placeholder="留空自动生成",
        help="盘中快照可填 2026-04-20 09:35:00；留空时盘后默认 15:00。",
    )

    if st.sidebar.button("运行日常主流程", type="primary", width="stretch"):
        try:
            with st.spinner("正在运行 Top100 主流程..."):
                result = run_pipeline(
                    native_fetch=not recompute_only,
                    capture_type=CAPTURE_TYPE_LABELS[capture_type_label],
                    snapshot_time=snapshot_time or None,
                )
            st.session_state["last_pipeline_result"] = result
            data_status = str((result.get("data", {}) or {}).get("status", ""))
            data_reason = str((result.get("data", {}) or {}).get("reason", "") or "").strip()
            if data_status == "skipped_market_closed":
                st.sidebar.warning(data_reason or "????????????????")
            elif data_status == "skipped_existing_snapshot":
                st.sidebar.info(data_reason or "????????????????")
            elif data_status == "stale_settlement":
                st.sidebar.warning(data_reason or "???????????????? stale_settlement?")
            else:
                st.sidebar.success("??????")
        except Exception as exc:
            st.session_state["last_pipeline_error"] = str(exc)
            st.sidebar.error(f"??????{exc}")

    if "last_pipeline_result" in st.session_state:
        result = st.session_state["last_pipeline_result"]
        with st.sidebar.expander("最近一次结果", expanded=False):
            st.write(f"开始：{result.get('started_at', '-')}")
            st.write(f"结束：{result.get('finished_at', '-')}")
            data_stats = result.get("data", {}) or {}
            feature_stats = result.get("features", {}) or {}
            signal_stats = result.get("signals", {}) or {}
            report_stats = result.get("reports", {}) or {}
            freshness_stats = result.get("freshness", {}) or {}
            st.write(f"Top100 样本：{data_stats.get('popularity_rows', '-')}")
            if data_stats.get("reason"):
                st.warning(str(data_stats.get("reason")))
            if freshness_stats:
                freshness_text = "通过" if freshness_stats.get("is_fresh") else "未通过"
                st.write(f"数据新鲜度：{freshness_text}")
                freshness_summary_text = str(freshness_stats.get("summary", "") or "").strip()
                if freshness_summary_text:
                    st.caption(freshness_summary_text)
            st.write(f"特征行数：{feature_stats.get('rows', '-')}")
            st.write(f"推送行数：{signal_stats.get('pushed_rows', '-')}")
            st.write(f"强势复盘：{report_stats.get('strong_recap_rows', '-')}")
            strategy_runs = result.get("strategies", {}) or {}
            strategy_order = result.get("strategy_versions", []) or []
            if strategy_runs:
                st.write("策略版本：")
                for version in strategy_order:
                    version_stats = strategy_runs.get(version, {}) or {}
                    version_signal_stats = version_stats.get("signals", {}) or {}
                    version_report_stats = version_stats.get("reports", {}) or {}
                    version_name = get_strategy_profile(version).get("name", version.upper())
                    st.write(
                        f"{version_name}：推送 {version_signal_stats.get('pushed_rows', '-')}，"
                        f"强势复盘 {version_report_stats.get('strong_recap_rows', '-')}"
                    )

    if "last_pipeline_error" in st.session_state:
        with st.sidebar.expander("最近一次错误", expanded=False):
            st.code(st.session_state["last_pipeline_error"])

    with st.sidebar.expander("固定说明", expanded=False):
        st.markdown("**日常主流程**")
        for step_no, title, description in DAILY_FLOW_STEPS:
            st.markdown(f"**{step_no}. {title}**")
            st.caption(description)
        st.markdown("**使用建议**")
        st.caption("每天主要看：今日快照状态、云端同步结果、最近一次结果。固定说明不用天天展开。")

    return selected_strategy_version


selected_strategy_version = render_sidebar_daily_flow()

data = load_all(selected_strategy_version)
signals_df = data["signals"]
followups_df = data["followups"]
backtest_summary_df = normalize_backtest_summary(data.get("backtest_summary", pd.DataFrame()), selected_strategy_version)
latest_push_df = data["latest_push"]
fast_strategy_df = data["fast_strategy"]
fast_strategy_audit_df = data["fast_strategy_audit"]
strong_recap_df = data["strong_recap"]
rule_eval_df = (
    build_backtest_metric_matrix(
        backtest_summary_df,
        strategy_version=selected_strategy_version,
        metric_keys=list(BACKTEST_METRIC_KEY_TO_LABEL.keys()),
    )
    if not backtest_summary_df.empty
    else data["rule_eval"]
)
lesson_eval_df = data["lesson_eval"]
market_regime_df = data["market_regime"]
freshness_report = build_data_freshness_report(followups_df, market_regime_df)

if selected_strategy_version != DEFAULT_APP_STRATEGY_VERSION and signals_df.empty:
    st.warning(f"{get_strategy_profile(selected_strategy_version).get('name', selected_strategy_version.upper())} 还没有产出结果，先运行一次主流程即可生成。")

st.title("Top100 情绪动量系统")
st.caption("只研究每日人气 Top100：今天推谁、为什么推、后来涨得怎么样。")
st.caption(f"当前查看策略：{get_strategy_profile(selected_strategy_version).get('name', selected_strategy_version.upper())}")

latest_date = "-"
if not signals_df.empty and "signal_date" in signals_df.columns:
    latest_date = str(signals_df["signal_date"].dropna().astype(str).max())
sample_day_count = signals_df["signal_date"].nunique() if not signals_df.empty and "signal_date" in signals_df.columns else 0
latest_market_label = "-"
market_status_note = ""
if not market_regime_df.empty and "signal_date" in market_regime_df.columns:
    market_view_df = market_regime_df.copy()
    market_view_df["signal_date"] = market_view_df["signal_date"].dropna().astype(str)
    if latest_date != "-":
        market_view_df = market_view_df[market_view_df["signal_date"].eq(latest_date)]
    if market_view_df.empty:
        market_view_df = market_regime_df.copy()
    if not market_view_df.empty:
        market_row = market_view_df.sort_values("signal_date").iloc[-1]
        market_regime_text = str(market_row.get("market_regime", "-") or "-")
        market_price_date = str(market_row.get("market_price_date", "") or "").strip()
        market_lag_days = pd.to_numeric(pd.Series([market_row.get("market_lag_days")]), errors="coerce").iloc[0]
        if pd.notna(market_lag_days) and int(market_lag_days) > 0:
            latest_market_label = "待更新"
            if market_price_date:
                market_status_note = f"收盘环境缓存暂未追到 {latest_date}，当前停在 {market_price_date}。"
        else:
            latest_market_label = market_regime_text

render_compact_cards(
    [
        ("最新样本日", latest_date),
        ("样本日", sample_day_count),
        ("主线样本", len(signals_df)),
        ("推送 / 候选", f"{len(latest_push_df)} / {len(fast_strategy_df)}"),
        ("收盘环境", latest_market_label),
        (
            "历史最新均值",
            fmt_pct(pd.to_numeric(followups_df["latest_return_pct"], errors="coerce").mean()) if not followups_df.empty and "latest_return_pct" in followups_df.columns else "-",
        ),
    ],
    widths=[1.1, 0.75, 0.85, 1.0, 0.92, 1.0],
    label_font_size="0.82rem",
    value_font_size="1.34rem",
    min_height="4.45rem",
)
if market_status_note:
    st.caption(market_status_note)
freshness_summary = str(freshness_report.get("summary", "") or "").strip()
if freshness_summary:
    if freshness_report.get("is_fresh"):
        st.caption(freshness_summary)
    else:
        st.warning(freshness_summary)
active_view = st.radio(
    "查看页面",
    options=["今日决策", "历史审查", "样本追踪", "强势复盘", "规则评估"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_active_view",
)

if active_view == "今日决策":
    st.subheader("今日决策")
    if fast_strategy_df.empty:
        st.info("暂无快策略数据。先运行 `python daily_job.py`。")
    else:
        summary_row = fast_strategy_df.iloc[0]
        render_compact_cards(
            [
                ("策略日", str(summary_row.get("strategy_date", "-"))),
                ("训练样本日", str(summary_row.get("training_date", "-"))),
                ("观察窗口", str(summary_row.get("analysis_window", "-"))),
                ("候选数", len(fast_strategy_df)),
            ],
            widths=[0.85, 1.0, 1.9, 0.7],
            label_font_size="0.82rem",
            value_font_size="1.34rem",
            min_height="4.45rem",
        )
        st.info(str(summary_row.get("learned_rule", "-")))
        st.markdown("##### 快策略候选池")
        fast_strategy_display_df = format_table_df(
            fast_strategy_df,
            columns=[
                "code",
                "name",
                "fast_score",
                "fast_level",
                "analysis_window",
                "fit_reasons",
                "next_session_plan",
            ],
            rename={
                "code": "代码",
                "name": "名称",
                "fast_score": "策略分",
                "fast_level": "快策略层级",
                "analysis_window": "观察窗口",
                "fit_reasons": "为什么入选",
                "next_session_plan": "下个交易日计划",
            },
        )
        st.dataframe(
            fast_strategy_display_df,
            width="stretch",
            hide_index=True,
            key="fast_strategy_table",
        )
        with st.expander("调试字段", expanded=False):
            display_table(
                fast_strategy_df,
                columns=[
                    "rank",
                    "code",
                    "name",
                    "emotion_score",
                    "market_regime",
                    "market_1d_pct",
                    "relative_1d_pct",
                    "market_adjustment",
                    "capture_type",
                    "snapshot_time",
                    "fit_reasons",
                    "reasons",
                    "risks",
                ],
                rename={
                    "rank": "人气排名",
                    "code": "代码",
                    "name": "名称",
                    "emotion_score": "情绪分",
                    "market_regime": "收盘环境",
                    "market_1d_pct": "大盘当日",
                    "relative_1d_pct": "跑赢大盘",
                    "market_adjustment": "大盘修正",
                    "capture_type": "采集类型",
                    "snapshot_time": "快照时间",
                    "fit_reasons": "快策略命中",
                    "reasons": "原始原因",
                    "risks": "风险",
                },
                limit=300,
            )

    st.divider()
    st.markdown("##### 今日推送执行单")
    if latest_push_df.empty:
        st.info("暂无今日推送数据。")
    else:
        execution_df = build_execution_display_df(latest_push_df, fast_strategy_df, selected_strategy_version)
        execution_scope = st.radio(
            "执行单范围",
            options=["优先关注", "强推/重点", "全部"],
            index=0,
            horizontal=True,
            key="latest_push_execution_scope",
        )
        st.caption("优先关注 = 候选池命中且推送层级为强推/重点；强推/重点 = 只按原始推送层级看。")
        visible_execution_df = execution_df
        if execution_scope == "优先关注":
            visible_execution_df = execution_df[execution_df["是否优先关注"].eq("是")].copy()
        elif execution_scope == "强推/重点":
            visible_execution_df = execution_df[execution_df["推送层级"].astype(str).isin(["强推观察", "重点观察"])].copy()

        if visible_execution_df.empty:
            st.info("当前范围暂无股票。")
        else:
            edited_execution_df = st.data_editor(
                visible_execution_df,
                width="stretch",
                hide_index=True,
                disabled=[column for column in visible_execution_df.columns if column != "查看"],
                column_config={
                    "查看": st.column_config.CheckboxColumn("查看", help="勾选后打开股票详情", width="small"),
                    "股票": st.column_config.TextColumn("股票", width="medium"),
                    "推送层级": st.column_config.TextColumn("推送层级", width="small"),
                    "当日涨跌": st.column_config.TextColumn("当日涨跌", width="small"),
                    "关键理由": st.column_config.TextColumn("关键理由", width="large"),
                    "候选池命中": st.column_config.TextColumn("候选池命中", width="small"),
                    "是否优先关注": st.column_config.TextColumn("是否优先关注", width="small"),
                },
                key=f"latest_push_execution_editor_{execution_scope}_{st.session_state.get(f'latest_push_execution_editor_reset_nonce_{execution_scope}', 0)}",
            )
            checked_indices = set(edited_execution_df.index[edited_execution_df["查看"].fillna(False)])
            previous_checked_indices = st.session_state.get("latest_push_checked_indices", set())
            newly_checked_indices = checked_indices - previous_checked_indices
            selected_execution_index = None
            if newly_checked_indices:
                selected_execution_index = sorted(newly_checked_indices)[-1]
            st.session_state["latest_push_checked_indices"] = checked_indices

            if selected_execution_index is not None:
                push_row = latest_push_df.reset_index(drop=True).iloc[int(selected_execution_index)]
                detail_row = build_detail_row(push_row, fast_strategy_df)
                default_detail_date = str(detail_row.get("strategy_date", detail_row.get("signal_date", latest_date)))
                render_stock_detail_dialog(detail_row, default_trade_date=default_detail_date)
                st.session_state[f"latest_push_execution_editor_reset_nonce_{execution_scope}"] = (
                    int(st.session_state.get(f"latest_push_execution_editor_reset_nonce_{execution_scope}", 0)) + 1
                )
                st.session_state["latest_push_checked_indices"] = set()

elif active_view == "历史审查":
        st.subheader("历史审查")
        st.caption("单独回看快策略结算结果，和今日执行页分开，避免塞在同一个页签里。")
        if fast_strategy_audit_df.empty:
            st.info("暂无审查记录。当前策略已记录后，等待后续行情更新。")
        else:
            audit_dates = sorted(fast_strategy_audit_df["strategy_date"].dropna().astype(str).unique(), reverse=True)
            audit_filter_cols = st.columns(3, gap="small")
            with audit_filter_cols[0]:
                st.caption("审查哪一天的策略")
                selected_audit_date = st.selectbox(
                    "审查哪一天的策略",
                    options=["全部"] + audit_dates,
                    index=0,
                    label_visibility="collapsed",
                    key="audit_selected_date",
                )
            with audit_filter_cols[1]:
                st.caption("审查范围")
                audit_scope = st.selectbox(
                    "审查范围",
                    options=["全部"] + sorted(fast_strategy_audit_df["audit_scope"].dropna().astype(str).unique()),
                    index=0,
                    label_visibility="collapsed",
                    key="audit_scope",
                )
            with audit_filter_cols[2]:
                st.caption("排序方式")
                audit_sort = st.selectbox(
                    "排序方式",
                    options=["复盘优先", "策略分最高", "至今涨幅最高", "人气排名靠前"],
                    index=0,
                    label_visibility="collapsed",
                    key="audit_sort",
                )
            audit_df = fast_strategy_audit_df.copy()
            if selected_audit_date != "全部":
                audit_df = audit_df[audit_df["strategy_date"].astype(str).eq(selected_audit_date)]
            if audit_scope != "全部":
                audit_df = audit_df[audit_df["audit_scope"].astype(str).eq(audit_scope)]
            if not audit_df.empty:
                audit_df = audit_df.copy()
                audit_df["fast_score"] = pd.to_numeric(audit_df.get("fast_score"), errors="coerce")
                audit_df["latest_return_pct"] = pd.to_numeric(audit_df.get("latest_return_pct"), errors="coerce")
                audit_df["rank"] = pd.to_numeric(audit_df.get("rank"), errors="coerce")
                if audit_sort == "策略分最高":
                    audit_df = audit_df.sort_values(["strategy_date", "fast_score", "rank"], ascending=[False, False, True], na_position="last")
                elif audit_sort == "至今涨幅最高":
                    audit_df = audit_df.sort_values(["strategy_date", "latest_return_pct", "fast_score", "rank"], ascending=[False, False, False, True], na_position="last")
                elif audit_sort == "人气排名靠前":
                    audit_df = audit_df.sort_values(["strategy_date", "rank", "fast_score"], ascending=[False, True, False], na_position="last")
            display_table(
                audit_df,
                columns=[
                    "strategy_date",
                    "audit_status",
                    "audit_scope",
                    "audit_result",
                    "lesson_type",
                    "rank",
                    "code",
                    "name",
                    "fast_level",
                    "fast_score",
                    "observed_days",
                    "tail_next_open_pct",
                    "tail_next_close_pct",
                    "latest_return_pct",
                    "return_3d_pct",
                    "return_5d_pct",
                    "max_gain_5d_pct",
                    "lesson_note",
                    "reasons",
                    "risks",
                ],
                rename={
                    "strategy_date": "策略日",
                    "audit_status": "审查状态",
                    "audit_scope": "审查范围",
                    "audit_result": "审查结果",
                    "lesson_type": "反哺标签",
                    "rank": "人气排名",
                    "code": "代码",
                    "name": "名称",
                    "fast_level": "快策略层级",
                    "fast_score": "快策略分",
                    "observed_days": "已观察天数",
                    "tail_next_open_pct": "次日开盘收益%",
                    "tail_next_close_pct": "次日收盘收益%",
                    "latest_return_pct": "至今涨跌%",
                    "return_3d_pct": "3日收益%",
                    "return_5d_pct": "5日收益%",
                    "max_gain_5d_pct": "5日最大上涨%",
                    "lesson_note": "反哺记录",
                    "reasons": "当时原因",
                    "risks": "当时风险",
                },
                column_config={
                    "审查结果": st.column_config.TextColumn(
                        "审查结果",
                        help="走势结论：根据已观察天数和当前后验收益判断这条样本到目前表现如何。",
                    ),
                    "反哺标签": st.column_config.TextColumn(
                        "反哺标签",
                        help="学习分桶：用于后续规则复盘和权重调整，不等同于走势结论。",
                    ),
                    "快策略分": st.column_config.TextColumn(
                        "快策略分",
                        help="反哺样本，不参与快策略打分。",
                    )
                },
                limit=300,
            )

elif active_view == "样本追踪":
    st.subheader("样本追踪")
    st.caption("每一行是某天进入 Top100 的一条样本，后面会持续结算 1/3/5/10 日。")
    if followups_df.empty:
        st.info("暂无后验数据。")
    else:
        working_df = followups_df.copy()
        dates = sorted(working_df["signal_date"].dropna().astype(str).unique(), reverse=True)
        selected_date = st.selectbox("查看哪一天的样本", options=["全部"] + dates, index=0)
        only_pushed = st.checkbox("只看当时被推送的样本", value=False)
        keyword = st.text_input("搜索代码或名称", value="")
        if selected_date != "全部":
            working_df = working_df[working_df["signal_date"].astype(str).eq(selected_date)]
        if only_pushed and "is_pushed" in working_df.columns:
            working_df = working_df[working_df["is_pushed"].astype(str).str.lower().isin(["true", "1"])]
        if keyword:
            mask = working_df["code"].astype(str).str.contains(keyword, case=False, na=False) | working_df["name"].astype(str).str.contains(keyword, case=False, na=False)
            working_df = working_df[mask]
        display_table(
            working_df.sort_values(["signal_date", "emotion_score"], ascending=[False, False], na_position="last"),
            columns=[
                "signal_date",
                "rank",
                "code",
                "name",
                "push_level",
                "emotion_score",
                "observed_days",
                "capture_type",
                "snapshot_time",
                "tail_next_open_pct",
                "tail_next_close_pct",
                "latest_return_pct",
                "return_3d_pct",
                "return_5d_pct",
                "max_gain_5d_pct",
                "reasons",
                "risks",
            ],
            rename={
                "signal_date": "加入日期",
                "rank": "人气排名",
                "code": "代码",
                "name": "名称",
                "push_level": "推送层级",
                "emotion_score": "情绪分",
                "observed_days": "已观察天数",
                "capture_type": "采集类型",
                "snapshot_time": "快照时间",
                "tail_next_open_pct": "次日开盘收益%",
                "tail_next_close_pct": "次日收盘收益%",
                "latest_return_pct": "至今涨跌%",
                "return_3d_pct": "3日收益%",
                "return_5d_pct": "5日收益%",
                "max_gain_5d_pct": "5日最大上涨%",
                "reasons": "当时原因",
                "risks": "当时风险",
            },
            limit=300,
        )

elif active_view == "强势复盘":
    st.subheader("强势复盘")
    st.caption("这里专门看后来涨得好的样本，用来反推什么特征更像圣阳股份。")
    if isinstance(freshness_report, dict) and not freshness_report.get("is_fresh"):
        st.warning(f"强势复盘当前还没通过数据新鲜度校验：{freshness_report.get('summary') or freshness_report.get('reason') or '请先跑完整收盘版主流程'}")
    display_table(
        strong_recap_df,
        columns=[
            "signal_date",
            "rank",
            "code",
            "name",
            "push_level",
            "emotion_score",
            "observed_days",
            "tail_next_open_pct",
            "tail_next_close_pct",
            "tail_next_max_gain_pct",
            "latest_return_pct",
            "best_return_available",
            "return_3d_pct",
            "return_5d_pct",
            "max_gain_5d_pct",
            "reasons",
            "risks",
        ],
        rename={
            "signal_date": "加入日期",
            "rank": "人气排名",
            "code": "代码",
            "name": "名称",
            "push_level": "当时层级",
            "emotion_score": "当时情绪分",
            "observed_days": "已观察天数",
            "tail_next_open_pct": "次日开盘收益%",
            "tail_next_close_pct": "次日收盘收益%",
            "tail_next_max_gain_pct": "次日最大冲高%",
            "latest_return_pct": "至今涨跌%",
            "best_return_available": "阶段最好表现%",
            "return_3d_pct": "3日收益%",
            "return_5d_pct": "5日收益%",
            "max_gain_5d_pct": "5日最大上涨%",
            "reasons": "当时原因",
            "risks": "当时风险",
        },
        limit=200,
    )

else:
    st.subheader("规则评估")
    st.caption("按推送层级、排名段、上榜阶段看后验表现。样本多起来后，这里就是调参依据。")
    compare_data = {
        version: (data if version == selected_strategy_version else load_all(version))
        for version in AVAILABLE_STRATEGY_VERSIONS
    }
    for version in AVAILABLE_STRATEGY_VERSIONS:
        version_backtest_summary = normalize_backtest_summary(compare_data.get(version, {}).get("backtest_summary", pd.DataFrame()), version)
        if version_backtest_summary.empty:
            continue
        compare_data[version]["backtest_summary"] = version_backtest_summary
        compare_data[version]["rule_eval"] = build_backtest_metric_matrix(
            version_backtest_summary,
            strategy_version=version,
            metric_keys=list(BACKTEST_METRIC_KEY_TO_LABEL.keys()),
        )
    compare_versions = [
        version
        for version in AVAILABLE_STRATEGY_VERSIONS
        if any(
            not compare_data.get(version, {}).get(dataset_key, pd.DataFrame()).empty
            for dataset_key in ["followups", "backtest_summary", "rule_eval"]
        )
    ]

    metric_options = [label for label in RETURN_METRIC_SPECS.keys() if label != "10日收益"]
    preferred_metric = strategy_default_metric_label(selected_strategy_version)
    default_metric = preferred_metric if preferred_metric in metric_options else ("5日收益" if "5日收益" in metric_options else metric_options[0])
    selected_perf_metric = default_metric

    if len(compare_versions) >= 2:
        baseline_version = compare_versions[0]
        st.markdown("##### 策略版本单日分层对比")
        compare_followups = {
            version: compare_data.get(version, {}).get("followups", pd.DataFrame())
            for version in compare_versions
        }
        performance_dates = sorted(
            {
                str(date)
                for frame in compare_followups.values()
                if not frame.empty and "signal_date" in frame.columns
                for date in frame["signal_date"].dropna().astype(str).unique()
            },
            reverse=True,
        )
        if not performance_dates:
            st.info("当前版本还没有可用于对比的后验数据。")
        else:
            default_perf_index = performance_dates.index(latest_date) if latest_date in performance_dates else 0
            control_cols = st.columns([1.1, 0.95, 1.05], gap="small")
            with control_cols[0]:
                selected_perf_date = st.selectbox("样本日期", options=performance_dates, index=default_perf_index, key="compare_push_level_perf_date")
            with control_cols[1]:
                selected_perf_metric = st.selectbox("收益窗口", options=metric_options, index=metric_options.index(default_metric), key="compare_push_level_perf_metric")
            with control_cols[2]:
                chart_mode = st.selectbox("图表类型", options=["柱状图", "日期折线图", "都显示"], index=2, key="compare_push_level_perf_chart_mode")
            selected_perf_metric_key = RULE_EVAL_METRIC_LABEL_TO_KEY.get(selected_perf_metric, "metric")

            summary_by_version = {
                version: summarize_push_level_performance(compare_followups.get(version, pd.DataFrame()), selected_perf_date, selected_perf_metric)
                for version in compare_versions
            }
            compare_cols = st.columns(len(compare_versions), gap="large")
            for version, column in zip(compare_versions, compare_cols):
                with column:
                    st.markdown(f"**{version_display_name(version)}**")
                    version_summary_df = summary_by_version.get(version, pd.DataFrame())
                    settled_total = int(pd.to_numeric(version_summary_df.get("valid_count"), errors="coerce").fillna(0).sum()) if not version_summary_df.empty else 0
                    pending_total = int(pd.to_numeric(version_summary_df.get("pending_count"), errors="coerce").fillna(0).sum()) if not version_summary_df.empty else 0
                    up_total = int(pd.to_numeric(version_summary_df.get("up_count"), errors="coerce").fillna(0).sum()) if not version_summary_df.empty else 0
                    render_compact_cards(
                        [
                            ("层级数", len(version_summary_df)),
                            ("已结算", settled_total),
                            ("待结算", pending_total),
                            ("上涨数", up_total),
                        ],
                        widths=[0.9, 0.9, 0.9, 0.9],
                        label_font_size="0.78rem",
                        value_font_size="1.02rem",
                        min_height="3.85rem",
                    )
                    if chart_mode in {"柱状图", "都显示"}:
                        st.plotly_chart(
                            build_push_level_bar_figure(version_summary_df, selected_perf_metric, selected_perf_date),
                            key=f"compare_bar_{version}_{selected_perf_date}_{selected_perf_metric_key}",
                            width="stretch",
                            config={"displayModeBar": False},
                        )

            st.caption(f"下面这张表以 {version_display_name(baseline_version)} 为基准，变化列按“当前版本 - 基准版本”计算。")
            push_level_compare_df = build_push_level_compare_table(summary_by_version)
            push_level_compare_columns = ["push_level"]
            push_level_compare_rename = {"push_level": "推送层级"}
            for version in compare_versions:
                push_level_compare_columns.extend(
                    [
                        f"{version}_sample_count",
                        f"{version}_valid_count",
                        f"{version}_win_rate_pct",
                        f"{version}_avg_return_pct",
                    ]
                )
                version_name = version_display_name(version)
                push_level_compare_rename[f"{version}_sample_count"] = f"{version_name}样本数"
                push_level_compare_rename[f"{version}_valid_count"] = f"{version_name}已结算"
                push_level_compare_rename[f"{version}_win_rate_pct"] = f"{version_name}上涨胜率%"
                push_level_compare_rename[f"{version}_avg_return_pct"] = f"{version_name}平均收益%"
            for version in compare_versions[1:]:
                push_level_compare_columns.extend(
                    [
                        f"{version}_vs_{baseline_version}_win_rate_delta_pct",
                        f"{version}_vs_{baseline_version}_avg_return_delta_pct",
                        f"{version}_vs_{baseline_version}_up_count_delta",
                    ]
                )
                version_name = version_display_name(version)
                base_name = version_display_name(baseline_version)
                push_level_compare_rename[f"{version}_vs_{baseline_version}_win_rate_delta_pct"] = f"{version_name}-{base_name}胜率变化"
                push_level_compare_rename[f"{version}_vs_{baseline_version}_avg_return_delta_pct"] = f"{version_name}-{base_name}均值变化"
                push_level_compare_rename[f"{version}_vs_{baseline_version}_up_count_delta"] = f"{version_name}-{base_name}上涨数变化"
            display_table(
                push_level_compare_df,
                columns=push_level_compare_columns,
                rename=push_level_compare_rename,
            )

            if chart_mode in {"日期折线图", "都显示"}:
                st.markdown("##### 按样本日期趋势对比")
                st.caption("这块更适合看阶段性稳定度：同一层级的胜率和平均收益是否随着样本日期出现明显起伏。")
                trend_scope = st.selectbox(
                    "折线图范围",
                    options=["最近 5 个样本日", "最近 10 个样本日", "全部样本日"],
                    index=1 if len(performance_dates) > 5 else 2,
                    key="compare_push_level_trend_scope",
                )
                if trend_scope == "最近 5 个样本日":
                    trend_dates = sorted(performance_dates[:5])
                elif trend_scope == "最近 10 个样本日":
                    trend_dates = sorted(performance_dates[:10])
                else:
                    trend_dates = sorted(performance_dates)

                for version in compare_versions:
                    trend_df = summarize_push_level_trend(
                        compare_followups.get(version, pd.DataFrame()),
                        selected_perf_metric,
                        signal_dates=trend_dates,
                    )
                    st.plotly_chart(
                        build_push_level_trend_figure(
                            trend_df,
                            selected_perf_metric,
                            f"{version_display_name(version)}样本日期趋势",
                        ),
                        key=f"compare_trend_{version}_{selected_perf_metric_key}_{trend_scope}",
                        width="stretch",
                        config={"displayModeBar": False},
                    )

        st.divider()
        st.markdown("##### 策略版本汇总规则对比")
        st.markdown("##### 正式回测汇总")
        st.caption("这里直接读取 `backtest_summary*.csv`，按正式回测汇总看版本差异；下面保留的宽表则是从这套正式汇总即时透视出来的兼容视图。")
        compare_backtest_summary = {
            version: compare_data.get(version, {}).get("backtest_summary", pd.DataFrame())
            for version in compare_versions
        }
        summary_versions = [version for version in compare_versions if not compare_backtest_summary.get(version, pd.DataFrame()).empty]
        if summary_versions:
            summary_metric_options = [
                (metric_key, metric_label)
                for metric_key, metric_label in BACKTEST_METRIC_KEY_TO_LABEL.items()
                if any(compare_backtest_summary.get(version, pd.DataFrame())["metric_key"].eq(metric_key).any() for version in summary_versions)
            ]
            summary_group_options = [
                group_name
                for group_name in BACKTEST_GROUP_NAMES
                if any(compare_backtest_summary.get(version, pd.DataFrame())["group_name"].eq(group_name).any() for version in summary_versions)
            ]
            default_summary_metric_label = strategy_default_metric_label(selected_strategy_version)
            default_summary_metric_index = 0
            for index, (_, metric_label) in enumerate(summary_metric_options):
                if metric_label == default_summary_metric_label:
                    default_summary_metric_index = index
                    break

            summary_controls = st.columns([1.0, 1.0, 0.8, 0.8], gap="small")
            with summary_controls[0]:
                selected_summary_group_name = st.selectbox("汇总分组维度", options=summary_group_options, index=0, key="backtest_summary_group_name")
            with summary_controls[1]:
                selected_summary_metric_label = st.selectbox(
                    "汇总收益窗口",
                    options=[label for _, label in summary_metric_options],
                    index=default_summary_metric_index,
                    key="backtest_summary_metric_label",
                )
            with summary_controls[2]:
                summary_min_sample_count = int(st.number_input("最小样本数", min_value=0, max_value=9999, value=0, step=1, key="backtest_summary_min_sample"))
            with summary_controls[3]:
                summary_min_valid_count = int(st.number_input("最小已结算", min_value=0, max_value=9999, value=0, step=1, key="backtest_summary_min_valid"))

            selected_summary_metric_key = next(
                (metric_key for metric_key, metric_label in summary_metric_options if metric_label == selected_summary_metric_label),
                summary_metric_options[0][0],
            )
            backtest_snapshot_by_version = {
                version: build_backtest_metric_snapshot(
                    compare_backtest_summary.get(version, pd.DataFrame()),
                    metric_key=selected_summary_metric_key,
                    group_name=selected_summary_group_name,
                    strategy_version=version,
                    min_sample_count=summary_min_sample_count,
                    min_valid_count=summary_min_valid_count,
                )
                for version in summary_versions
            }

            snapshot_cols = st.columns(len(summary_versions), gap="large")
            for version, column in zip(summary_versions, snapshot_cols):
                with column:
                    st.markdown(f"**{version_display_name(version)}**")
                    version_snapshot_df = backtest_snapshot_by_version.get(version, pd.DataFrame())
                    settled_total = int(pd.to_numeric(version_snapshot_df.get("valid_count"), errors="coerce").fillna(0).sum()) if not version_snapshot_df.empty else 0
                    pushed_total = int(pd.to_numeric(version_snapshot_df.get("pushed_count"), errors="coerce").fillna(0).sum()) if not version_snapshot_df.empty else 0
                    avg_return_mean = pd.to_numeric(version_snapshot_df.get("avg_return_pct"), errors="coerce").dropna().mean() if not version_snapshot_df.empty else None
                    render_compact_cards(
                        [
                            ("分组数", len(version_snapshot_df)),
                            ("推送数", pushed_total),
                            ("已结算", settled_total),
                            ("组均值", "-" if avg_return_mean is None or pd.isna(avg_return_mean) else f"{float(avg_return_mean):.2f}%"),
                        ],
                        widths=[0.9, 0.9, 0.9, 1.0],
                        label_font_size="0.78rem",
                        value_font_size="1.02rem",
                        min_height="3.85rem",
                    )
                    st.plotly_chart(
                        build_backtest_group_bar_figure(
                            version_snapshot_df,
                            selected_summary_metric_label,
                            f"{version_display_name(version)} · {selected_summary_group_name}",
                        ),
                        key=f"backtest_summary_bar_{version}_{selected_summary_group_name}_{selected_summary_metric_key}",
                        width="stretch",
                        config={"displayModeBar": False},
                    )
            if len(summary_versions) >= 2:
                baseline_summary_version = summary_versions[0]
                st.caption(f"下面这张正式回测对比表以 {version_display_name(baseline_summary_version)} 为基准，变化列按“当前版本 - 基准版本”计算。")
                backtest_compare_df = build_backtest_compare_table(
                    compare_backtest_summary,
                    metric_key=selected_summary_metric_key,
                    group_name=selected_summary_group_name,
                    min_sample_count=summary_min_sample_count,
                    min_valid_count=summary_min_valid_count,
                )
                backtest_compare_columns = ["group_name", "group_value"]
                backtest_compare_rename = {
                    "group_name": "分组类型",
                    "group_value": "分组",
                }
                for version in summary_versions:
                    version_name = version_display_name(version)
                    backtest_compare_columns.extend(
                        [
                            f"{version}_sample_count",
                            f"{version}_pushed_count",
                            f"{version}_valid_count",
                            f"{version}_win_rate_pct",
                            f"{version}_avg_return_pct",
                            f"{version}_strong_rate_pct",
                        ]
                    )
                    backtest_compare_rename[f"{version}_sample_count"] = f"{version_name}样本数"
                    backtest_compare_rename[f"{version}_pushed_count"] = f"{version_name}推送数"
                    backtest_compare_rename[f"{version}_valid_count"] = f"{version_name}已结算"
                    backtest_compare_rename[f"{version}_win_rate_pct"] = f"{version_name}{selected_summary_metric_label}胜率%"
                    backtest_compare_rename[f"{version}_avg_return_pct"] = f"{version_name}{selected_summary_metric_label}均值%"
                    backtest_compare_rename[f"{version}_strong_rate_pct"] = f"{version_name}{selected_summary_metric_label}强势率%"
                for version in summary_versions[1:]:
                    version_name = version_display_name(version)
                    base_name = version_display_name(baseline_summary_version)
                    backtest_compare_columns.extend(
                        [
                            f"{version}_vs_{baseline_summary_version}_pushed_count_delta",
                            f"{version}_vs_{baseline_summary_version}_valid_count_delta",
                            f"{version}_vs_{baseline_summary_version}_win_rate_pct_delta",
                            f"{version}_vs_{baseline_summary_version}_avg_return_pct_delta",
                            f"{version}_vs_{baseline_summary_version}_strong_rate_pct_delta",
                        ]
                    )
                    backtest_compare_rename[f"{version}_vs_{baseline_summary_version}_pushed_count_delta"] = f"{version_name}-{base_name}推送变化"
                    backtest_compare_rename[f"{version}_vs_{baseline_summary_version}_valid_count_delta"] = f"{version_name}-{base_name}结算变化"
                    backtest_compare_rename[f"{version}_vs_{baseline_summary_version}_win_rate_pct_delta"] = f"{version_name}-{base_name}胜率变化"
                    backtest_compare_rename[f"{version}_vs_{baseline_summary_version}_avg_return_pct_delta"] = f"{version_name}-{base_name}均值变化"
                    backtest_compare_rename[f"{version}_vs_{baseline_summary_version}_strong_rate_pct_delta"] = f"{version_name}-{base_name}强势率变化"
                display_table(
                    backtest_compare_df,
                    columns=backtest_compare_columns,
                    rename=backtest_compare_rename,
                    limit=40,
                )
        else:
            st.info("当前还没有可用的正式回测汇总。")

        summary_metric_key = RULE_EVAL_METRIC_LABEL_TO_KEY.get(selected_perf_metric, "5d")
        rule_eval_compare_df = build_rule_eval_compare_table(
            {
                version: compare_data.get(version, {}).get("rule_eval", pd.DataFrame())
                for version in compare_versions
            },
            summary_metric_key,
        )
        rule_eval_compare_columns = ["group_name", "group_value"]
        rule_eval_compare_rename = {
            "group_name": "分组类型",
            "group_value": "分组",
        }
        for version in compare_versions:
            rule_eval_compare_columns.extend(
                [
                    f"{version}_pushed_count",
                    f"{version}_win_rate_{summary_metric_key}",
                    f"{version}_avg_{summary_metric_key}",
                ]
            )
            version_name = version_display_name(version)
            rule_eval_compare_rename[f"{version}_pushed_count"] = f"{version_name}推送数"
            rule_eval_compare_rename[f"{version}_win_rate_{summary_metric_key}"] = f"{version_name}{selected_perf_metric}胜率%"
            rule_eval_compare_rename[f"{version}_avg_{summary_metric_key}"] = f"{version_name}{selected_perf_metric}均值%"
        for version in compare_versions[1:]:
            version_name = version_display_name(version)
            base_name = version_display_name(baseline_version)
            rule_eval_compare_columns.extend(
                [
                    f"{version}_vs_{baseline_version}_pushed_count_delta",
                    f"{version}_vs_{baseline_version}_win_rate_{summary_metric_key}_delta",
                    f"{version}_vs_{baseline_version}_avg_{summary_metric_key}_delta",
                ]
            )
            rule_eval_compare_rename[f"{version}_vs_{baseline_version}_pushed_count_delta"] = f"{version_name}-{base_name}推送变化"
            rule_eval_compare_rename[f"{version}_vs_{baseline_version}_win_rate_{summary_metric_key}_delta"] = f"{version_name}-{base_name}{selected_perf_metric}胜率变化"
            rule_eval_compare_rename[f"{version}_vs_{baseline_version}_avg_{summary_metric_key}_delta"] = f"{version_name}-{base_name}{selected_perf_metric}均值变化"
        display_table(
            rule_eval_compare_df,
            columns=rule_eval_compare_columns,
            rename=rule_eval_compare_rename,
            limit=30,
        )
        st.divider()

    st.markdown(f"##### 当前版本详细表：{version_display_name(selected_strategy_version)}")
    if selected_strategy_version == "v3":
        rule_eval_columns = [
            "group_name",
            "group_value",
            "sample_count",
            "pushed_count",
            "valid_tail_next_open",
            "avg_tail_next_open",
            "win_rate_tail_next_open",
            "valid_tail_next_close",
            "avg_tail_next_close",
            "win_rate_tail_next_close",
            "valid_3d",
            "avg_3d",
            "win_rate_3d",
        ]
        rule_eval_rename = {
            "group_name": "分组类型",
            "group_value": "分组",
            "sample_count": "样本数",
            "pushed_count": "推送数",
            "valid_tail_next_open": "次日开盘已结算",
            "avg_tail_next_open": "次日开盘均值%",
            "win_rate_tail_next_open": "次日开盘胜率%",
            "valid_tail_next_close": "次日收盘已结算",
            "avg_tail_next_close": "次日收盘均值%",
            "win_rate_tail_next_close": "次日收盘胜率%",
            "valid_3d": "3日已结算",
            "avg_3d": "3日均值%",
            "win_rate_3d": "3日胜率%",
        }
    else:
        rule_eval_columns = [
            "group_name",
            "group_value",
            "sample_count",
            "pushed_count",
            "valid_3d",
            "avg_3d",
            "win_rate_3d",
            "strong_rate_3d",
            "valid_5d",
            "avg_5d",
            "win_rate_5d",
            "strong_rate_5d",
        ]
        rule_eval_rename = {
            "group_name": "分组类型",
            "group_value": "分组",
            "sample_count": "样本数",
            "pushed_count": "推送数",
            "valid_3d": "已结算3日",
            "avg_3d": "3日均值%",
            "win_rate_3d": "3日胜率%",
            "strong_rate_3d": "3日强势率%",
            "valid_5d": "已结算5日",
            "avg_5d": "5日均值%",
            "win_rate_5d": "5日胜率%",
            "strong_rate_5d": "5日强势率%",
        }
    display_table(
        rule_eval_df,
        columns=rule_eval_columns,
        rename=rule_eval_rename,
    )
    with st.expander("查看当前版本正式回测原始汇总行", expanded=False):
        display_table(
            backtest_summary_df,
            columns=[
                "strategy_version",
                "group_name",
                "group_value",
                "metric_key",
                "metric_label",
                "sample_count",
                "pushed_count",
                "valid_count",
                "avg_return_pct",
                "win_rate_pct",
                "strong_rate_pct",
                "generated_at",
            ],
            rename={
                "strategy_version": "版本",
                "group_name": "分组类型",
                "group_value": "分组",
                "metric_key": "指标键",
                "metric_label": "收益窗口",
                "sample_count": "样本数",
                "pushed_count": "推送数",
                "valid_count": "已结算",
                "avg_return_pct": "平均收益%",
                "win_rate_pct": "胜率%",
                "strong_rate_pct": "强势率%",
                "generated_at": "生成时间",
            },
            limit=120,
        )
    st.subheader(f"反哺统计：{version_display_name(selected_strategy_version)}")
    st.caption("这里看的是反哺标签本身的后验表现，用来判断哪些标签该加权，哪些该降权。")
    display_table(
        lesson_eval_df,
        columns=[
            "lesson_type",
            "sample_count",
            "valid_latest",
            "avg_latest_return_pct",
            "win_rate_latest_pct",
            "valid_3d",
            "avg_3d_return_pct",
            "win_rate_3d_pct",
            "valid_5d",
            "avg_5d_return_pct",
            "win_rate_5d_pct",
            "valid_max_gain_5d",
            "avg_max_gain_5d_pct",
            "strong_rate_5d_pct",
        ],
        rename={
            "lesson_type": "反哺标签",
            "sample_count": "样本数",
            "valid_latest": "有最新收益",
            "avg_latest_return_pct": "最新均值%",
            "win_rate_latest_pct": "最新胜率%",
            "valid_3d": "已结算3日",
            "avg_3d_return_pct": "3日均值%",
            "win_rate_3d_pct": "3日胜率%",
            "valid_5d": "已结算5日",
            "avg_5d_return_pct": "5日均值%",
            "win_rate_5d_pct": "5日胜率%",
            "valid_max_gain_5d": "有峰值收益",
            "avg_max_gain_5d_pct": "5日峰值均值%",
            "strong_rate_5d_pct": "5日强势率%",
        },
        limit=50,
    )
