from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.followups import build_followups


class FollowupsPriceCacheTest(unittest.TestCase):
    @patch("src.followups.load_price_data")
    def test_build_followups_reuses_price_data_for_same_code(self, mock_load_price_data) -> None:
        mock_load_price_data.return_value = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-04-22"), "open": 10, "close": 10, "high": 10, "low": 10, "volume": 1},
                {"date": pd.Timestamp("2026-04-23"), "open": 11, "close": 11, "high": 11.5, "low": 10.5, "volume": 1},
                {"date": pd.Timestamp("2026-04-24"), "open": 12, "close": 12, "high": 12.5, "low": 11.0, "volume": 1},
                {"date": pd.Timestamp("2026-04-27"), "open": 13, "close": 13, "high": 13.5, "low": 12.0, "volume": 1},
            ]
        )

        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-22", "code": "300067", "name": "安诺其", "close": 10, "price_date": "2026-04-22", "emotion_score": 50, "rank": 1},
                {"signal_date": "2026-04-23", "code": "300067", "name": "安诺其", "close": 11, "price_date": "2026-04-23", "emotion_score": 49, "rank": 2},
            ]
        )

        result = build_followups(signal_df, days=[1, 3, 5])

        self.assertEqual(mock_load_price_data.call_count, 1)
        self.assertEqual(len(result), 2)
        self.assertIn("return_3d_pct", result.columns)

    @patch("src.followups.load_price_data")
    def test_build_followups_adds_tail_next_day_metrics(self, mock_load_price_data) -> None:
        mock_load_price_data.return_value = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-04-22"), "open": 10, "close": 10, "high": 10, "low": 10, "volume": 1},
                {"date": pd.Timestamp("2026-04-23"), "open": 10.5, "close": 11.0, "high": 11.4, "low": 10.2, "volume": 1},
                {"date": pd.Timestamp("2026-04-24"), "open": 11.2, "close": 11.3, "high": 11.6, "low": 10.9, "volume": 1},
            ]
        )

        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-22", "code": "300067", "name": "安诺其", "close": 10, "price_date": "2026-04-22", "emotion_score": 50, "rank": 1},
            ]
        )

        result = build_followups(signal_df, days=[1, 3, 5])
        row = result.iloc[0]

        self.assertTrue(bool(row["settled_tail_next_day"]))
        self.assertAlmostEqual(float(row["tail_next_open_pct"]), 5.0, places=6)
        self.assertAlmostEqual(float(row["tail_next_close_pct"]), 10.0, places=6)
        self.assertAlmostEqual(float(row["tail_next_max_gain_pct"]), 14.0, places=6)
        self.assertAlmostEqual(float(row["tail_next_max_drawdown_pct"]), 2.0, places=6)

    @patch("src.followups.load_price_data")
    def test_build_followups_fills_missing_signal_price_from_price_history(self, mock_load_price_data) -> None:
        mock_load_price_data.return_value = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-05-21"), "open": 10, "close": 10, "high": 10, "low": 10, "volume": 1},
                {"date": pd.Timestamp("2026-05-22"), "open": 11, "close": 12, "high": 13, "low": 9, "volume": 1},
            ]
        )

        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-05-21", "code": "001259", "name": "利仁科技", "close": None, "price_date": None, "emotion_score": 0, "rank": 25},
            ]
        )

        result = build_followups(signal_df, days=[1])
        row = result.iloc[0]

        self.assertTrue(bool(row["settled_1d"]))
        self.assertEqual(float(row["signal_close"]), 10.0)
        self.assertEqual(row["latest_price_date"], "2026-05-22")
        self.assertAlmostEqual(float(row["return_1d_pct"]), 20.0, places=6)


if __name__ == "__main__":
    unittest.main()
