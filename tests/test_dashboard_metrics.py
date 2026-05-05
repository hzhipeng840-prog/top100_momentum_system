from __future__ import annotations

import unittest

import pandas as pd

from src.dashboard_metrics import summarize_push_level_performance, summarize_push_level_trend


class DashboardMetricsTest(unittest.TestCase):
    def test_summarize_push_level_performance_groups_by_date_and_level(self) -> None:
        followup_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-10", "push_level": "重点观察", "return_5d_pct": 5.0, "settled_5d": True},
                {"signal_date": "2026-04-10", "push_level": "重点观察", "return_5d_pct": -1.0, "settled_5d": True},
                {"signal_date": "2026-04-10", "push_level": "普通观察", "return_5d_pct": 2.0, "settled_5d": True},
                {"signal_date": "2026-04-10", "push_level": "不推送", "return_5d_pct": None, "settled_5d": False},
                {"signal_date": "2026-04-11", "push_level": "重点观察", "return_5d_pct": 9.0, "settled_5d": True},
            ]
        )

        result = summarize_push_level_performance(followup_df, "2026-04-10", "5日收益")

        focus_row = result[result["push_level"].eq("重点观察")].iloc[0]
        normal_row = result[result["push_level"].eq("普通观察")].iloc[0]
        ignored_row = result[result["push_level"].eq("不推送")].iloc[0]

        self.assertEqual(int(focus_row["sample_count"]), 2)
        self.assertEqual(int(focus_row["valid_count"]), 2)
        self.assertEqual(int(focus_row["up_count"]), 1)
        self.assertEqual(float(focus_row["win_rate_pct"]), 50.0)
        self.assertEqual(float(focus_row["avg_return_pct"]), 2.0)

        self.assertEqual(int(normal_row["sample_count"]), 1)
        self.assertEqual(float(normal_row["win_rate_pct"]), 100.0)
        self.assertEqual(float(normal_row["avg_return_pct"]), 2.0)

        self.assertEqual(int(ignored_row["valid_count"]), 0)
        self.assertEqual(int(ignored_row["pending_count"]), 1)
        self.assertTrue(pd.isna(ignored_row["win_rate_pct"]))

    def test_summarize_push_level_supports_tail_metrics(self) -> None:
        followup_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-10", "push_level": "强推观察", "tail_next_close_pct": 3.0, "settled_tail_next_day": True},
                {"signal_date": "2026-04-10", "push_level": "强推观察", "tail_next_close_pct": -1.0, "settled_tail_next_day": True},
                {"signal_date": "2026-04-10", "push_level": "普通观察", "tail_next_close_pct": None, "settled_tail_next_day": False},
            ]
        )

        result = summarize_push_level_performance(followup_df, "2026-04-10", "次日收盘收益")
        focus_row = result[result["push_level"].eq("强推观察")].iloc[0]

        self.assertEqual(int(focus_row["valid_count"]), 2)
        self.assertEqual(float(focus_row["avg_return_pct"]), 1.0)
        self.assertEqual(float(focus_row["win_rate_pct"]), 50.0)

    def test_summarize_push_level_trend_stacks_multiple_dates(self) -> None:
        followup_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-10", "push_level": "重点观察", "return_3d_pct": 5.0, "settled_3d": True},
                {"signal_date": "2026-04-10", "push_level": "重点观察", "return_3d_pct": -1.0, "settled_3d": True},
                {"signal_date": "2026-04-11", "push_level": "重点观察", "return_3d_pct": 4.0, "settled_3d": True},
                {"signal_date": "2026-04-11", "push_level": "普通观察", "return_3d_pct": 2.0, "settled_3d": True},
            ]
        )

        result = summarize_push_level_trend(
            followup_df,
            "3日收益",
            signal_dates=["2026-04-10", "2026-04-11"],
        )

        self.assertEqual(sorted(result["signal_date"].unique().tolist()), ["2026-04-10", "2026-04-11"])
        focus_latest = result[
            result["signal_date"].eq("2026-04-11") & result["push_level"].eq("重点观察")
        ].iloc[0]
        self.assertEqual(int(focus_latest["valid_count"]), 1)
        self.assertEqual(float(focus_latest["avg_return_pct"]), 4.0)


if __name__ == "__main__":
    unittest.main()
