from __future__ import annotations

import unittest

import pandas as pd

from src.backtest_queries import (
    build_backtest_compare_table,
    build_backtest_metric_matrix,
    build_backtest_metric_snapshot,
    normalize_backtest_summary,
    query_backtest_summary,
)


class BacktestQueriesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.summary_df = pd.DataFrame(
            [
                {
                    "strategy_version": "v1",
                    "group_name": "推送层级",
                    "group_value": "强推观察",
                    "metric_key": "3d",
                    "metric_label": "3日收益",
                    "sample_count": 10,
                    "pushed_count": 4,
                    "valid_count": 8,
                    "avg_return_pct": 5.2,
                    "win_rate_pct": 62.5,
                    "strong_rate_pct": 25.0,
                    "generated_at": "2026-05-06T12:00:00",
                },
                {
                    "strategy_version": "v1",
                    "group_name": "推送层级",
                    "group_value": "强推观察",
                    "metric_key": "5d",
                    "metric_label": "5日收益",
                    "sample_count": 10,
                    "pushed_count": 4,
                    "valid_count": 7,
                    "avg_return_pct": 7.5,
                    "win_rate_pct": 71.4,
                    "strong_rate_pct": 42.9,
                    "generated_at": "2026-05-06T12:00:00",
                },
                {
                    "strategy_version": "v1",
                    "group_name": "推送层级",
                    "group_value": "重点观察",
                    "metric_key": "3d",
                    "metric_label": "3日收益",
                    "sample_count": 12,
                    "pushed_count": 6,
                    "valid_count": 9,
                    "avg_return_pct": 2.1,
                    "win_rate_pct": 55.6,
                    "strong_rate_pct": 11.1,
                    "generated_at": "2026-05-06T12:00:00",
                },
                {
                    "strategy_version": "v2",
                    "group_name": "推送层级",
                    "group_value": "强推观察",
                    "metric_key": "3d",
                    "metric_label": "3日收益",
                    "sample_count": 11,
                    "pushed_count": 5,
                    "valid_count": 9,
                    "avg_return_pct": 6.0,
                    "win_rate_pct": 66.7,
                    "strong_rate_pct": 33.3,
                    "generated_at": "2026-05-06T12:05:00",
                },
            ]
        )

    def test_query_backtest_summary_filters_metric_and_thresholds(self) -> None:
        result = query_backtest_summary(
            self.summary_df,
            strategy_version="v1",
            metric_key="3d",
            group_name="推送层级",
            min_sample_count=11,
            min_valid_count=9,
        )

        self.assertEqual(result["group_value"].tolist(), ["重点观察"])
        self.assertEqual(result["metric_key"].tolist(), ["3d"])

    def test_build_backtest_metric_matrix_pivots_metrics_for_group(self) -> None:
        result = build_backtest_metric_matrix(
            self.summary_df,
            strategy_version="v1",
            group_name="推送层级",
            metric_keys=["3d", "5d"],
        )

        focus_row = result[result["group_value"].eq("强推观察")].iloc[0]
        self.assertEqual(int(focus_row["sample_count"]), 10)
        self.assertEqual(float(focus_row["avg_3d"]), 5.2)
        self.assertEqual(float(focus_row["win_rate_5d"]), 71.4)

    def test_build_backtest_compare_table_merges_versions_and_deltas(self) -> None:
        result = build_backtest_compare_table(
            {
                "v1": normalize_backtest_summary(self.summary_df[self.summary_df["strategy_version"].eq("v1")], "v1"),
                "v2": normalize_backtest_summary(self.summary_df[self.summary_df["strategy_version"].eq("v2")], "v2"),
            },
            metric_key="3d",
            group_name="推送层级",
        )

        focus_row = result[result["group_value"].eq("强推观察")].iloc[0]
        self.assertEqual(int(focus_row["v1_pushed_count"]), 4)
        self.assertEqual(int(focus_row["v2_pushed_count"]), 5)
        self.assertAlmostEqual(float(focus_row["v2_vs_v1_avg_return_pct_delta"]), 0.8, places=4)
        self.assertAlmostEqual(float(focus_row["v2_vs_v1_win_rate_pct_delta"]), 4.2, places=4)

    def test_build_backtest_metric_snapshot_returns_requested_metric_only(self) -> None:
        result = build_backtest_metric_snapshot(
            self.summary_df,
            strategy_version="v1",
            metric_key="5d",
            group_name="推送层级",
        )

        self.assertEqual(result["metric_key"].unique().tolist(), ["5d"])
        self.assertEqual(result["group_value"].tolist(), ["强推观察"])


if __name__ == "__main__":
    unittest.main()
