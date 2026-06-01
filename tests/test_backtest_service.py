from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.backtest_service import build_backtest_summary, build_rule_evaluation_view, run_backtest_service


class BacktestServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-10", "code": "000001", "push_level": "强推观察", "rank": 2, "consecutive_days": 1, "is_pushed": True},
                {"signal_date": "2026-04-10", "code": "000002", "push_level": "普通观察", "rank": 33, "consecutive_days": 2, "is_pushed": True},
            ]
        )
        self.followup_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-10",
                    "code": "000001",
                    "push_level": "强推观察",
                    "rank": 2,
                    "consecutive_days": 1,
                    "is_pushed": True,
                    "tail_next_close_pct": 4.0,
                    "settled_tail_next_day": True,
                    "return_3d_pct": 6.0,
                    "settled_3d": True,
                },
                {
                    "signal_date": "2026-04-10",
                    "code": "000002",
                    "push_level": "普通观察",
                    "rank": 33,
                    "consecutive_days": 2,
                    "is_pushed": True,
                    "tail_next_close_pct": -2.0,
                    "settled_tail_next_day": True,
                    "return_3d_pct": None,
                    "settled_3d": False,
                },
            ]
        )

    def test_build_backtest_summary_normalizes_window_group_records(self) -> None:
        summary_df = build_backtest_summary(self.signal_df, self.followup_df, strategy_version="v3", generated_at="2026-05-06T09:00:00")

        self.assertIn("metric_key", summary_df.columns)
        self.assertIn("group_name", summary_df.columns)
        self.assertIn("avg_return_pct", summary_df.columns)
        row = summary_df[
            summary_df["group_name"].eq("推送层级")
            & summary_df["group_value"].eq("强推观察")
            & summary_df["metric_key"].eq("tail_next_close")
        ].iloc[0]
        self.assertEqual(float(row["avg_return_pct"]), 4.0)
        self.assertEqual(float(row["win_rate_pct"]), 100.0)

    def test_build_rule_evaluation_view_keeps_legacy_wide_columns(self) -> None:
        summary_df = build_backtest_summary(self.signal_df, self.followup_df, strategy_version="v3", generated_at="2026-05-06T09:00:00")
        view_df = build_rule_evaluation_view(summary_df, strategy_version="v3")

        row = view_df[view_df["group_value"].eq("强推观察")].iloc[0]
        self.assertEqual(int(row["valid_tail_next_close"]), 1)
        self.assertEqual(float(row["avg_tail_next_close"]), 4.0)
        self.assertEqual(float(row["win_rate_tail_next_close"]), 100.0)

    def test_backtest_summary_prefers_latest_signal_push_state(self) -> None:
        signal_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-10",
                    "code": "000001",
                    "push_level": "观察池",
                    "rank": 2,
                    "consecutive_days": 1,
                    "is_pushed": False,
                }
            ]
        )
        followup_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-10",
                    "code": "000001",
                    "push_level": "强推观察",
                    "rank": 2,
                    "consecutive_days": 1,
                    "is_pushed": True,
                    "tail_next_close_pct": -6.0,
                    "settled_tail_next_day": True,
                }
            ]
        )

        summary_df = build_backtest_summary(signal_df, followup_df, strategy_version="v3", generated_at="2026-05-06T09:00:00")
        row = summary_df[
            summary_df["group_name"].eq("推送层级")
            & summary_df["group_value"].eq("观察池")
            & summary_df["metric_key"].eq("tail_next_close")
        ].iloc[0]

        self.assertEqual(int(row["pushed_count"]), 0)
        self.assertEqual(float(row["avg_return_pct"]), -6.0)

    def test_run_backtest_service_returns_both_summary_and_rule_eval(self) -> None:
        with patch("src.backtest_service.write_csv") as mock_write_csv:
            result = run_backtest_service(
                self.signal_df,
                self.followup_df,
                strategy_version="v3",
                summary_path=Path("backtest_summary_v3.csv"),
                rule_evaluation_path=Path("rule_evaluation_v3.csv"),
            )

        self.assertEqual(result.strategy_version, "v3")
        self.assertFalse(result.summary_df.empty)
        self.assertFalse(result.rule_evaluation_df.empty)
        self.assertEqual(mock_write_csv.call_count, 2)


if __name__ == "__main__":
    unittest.main()
