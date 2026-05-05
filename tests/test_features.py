from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.features import build_daily_features, select_strategy_snapshots


class FeatureSelectionTest(unittest.TestCase):
    def test_strategy_snapshot_priority_keeps_v1_post_close_and_v3_intraday(self) -> None:
        popularity_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "测试股",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "intraday_1430",
                    "snapshot_time": "2026-04-28 14:30:00",
                },
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "测试股",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-28 15:00:00",
                },
            ]
        )

        v1_df = select_strategy_snapshots(popularity_df, strategy_version="v1")
        v3_df = select_strategy_snapshots(popularity_df, strategy_version="v3")

        self.assertEqual(v1_df.iloc[0]["capture_type"], "post_close")
        self.assertEqual(v3_df.iloc[0]["capture_type"], "intraday_1430")

    @patch("src.features.load_intraday_snapshots")
    @patch("src.features.load_price_data")
    def test_v3_builds_intraday_features_from_matching_snapshot(
        self,
        mock_load_price_data,
        mock_load_intraday_snapshots,
    ) -> None:
        popularity_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "测试股",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "intraday_1430",
                    "snapshot_time": "2026-04-28 14:30:00",
                },
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "测试股",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-28 15:00:00",
                },
            ]
        )
        mock_load_intraday_snapshots.return_value = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-28",
                    "snapshot_time": "2026-04-28 14:30:00",
                    "capture_type": "intraday_1430",
                    "code": "600001",
                    "name": "测试股",
                    "last_price": 11.0,
                    "open": 10.3,
                    "prev_close": 10.0,
                    "current_return_pct": 10.0,
                    "day_high_so_far": 11.2,
                    "day_low_so_far": 10.2,
                    "volume_so_far": 160.0,
                    "amount_so_far": 1000.0,
                    "turnover_pct": 2.1,
                    "volume_ratio": 1.4,
                    "source": "unit-test",
                }
            ]
        )
        mock_load_price_data.return_value = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-04-21"), "open": 8.8, "close": 8.9, "high": 9.0, "low": 8.7, "volume": 100},
                {"date": pd.Timestamp("2026-04-22"), "open": 9.0, "close": 9.1, "high": 9.2, "low": 8.9, "volume": 105},
                {"date": pd.Timestamp("2026-04-23"), "open": 9.2, "close": 9.3, "high": 9.4, "low": 9.1, "volume": 110},
                {"date": pd.Timestamp("2026-04-24"), "open": 9.3, "close": 9.5, "high": 9.6, "low": 9.2, "volume": 120},
                {"date": pd.Timestamp("2026-04-25"), "open": 9.8, "close": 10.0, "high": 10.1, "low": 9.7, "volume": 130},
                {"date": pd.Timestamp("2026-04-27"), "open": 9.9, "close": 10.2, "high": 10.3, "low": 9.8, "volume": 140},
            ]
        )

        result = build_daily_features(popularity_df=popularity_df, strategy_version="v3")
        row = result.iloc[0]

        self.assertEqual(row["capture_type"], "intraday_1430")
        self.assertEqual(row["price_status"], "ok")
        self.assertEqual(row["price_date"], "2026-04-28")
        self.assertAlmostEqual(float(row["close"]), 11.0, places=6)
        self.assertAlmostEqual(float(row["day_return_pct"]), 10.0, places=6)
        self.assertEqual(int(row["price_lag_days"]), 0)


if __name__ == "__main__":
    unittest.main()
