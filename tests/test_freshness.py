from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.freshness import build_data_freshness_report


class FreshnessReportTest(unittest.TestCase):
    @patch("src.freshness.previous_a_share_trading_day", return_value=pd.Timestamp("2026-05-07"))
    @patch("src.freshness.latest_expected_market_date", return_value=pd.Timestamp("2026-05-08"))
    def test_build_data_freshness_report_detects_stale_settlement(self, _mock_latest_expected, _mock_previous_day) -> None:
        followup_df = pd.DataFrame(
            [
                {"signal_date": "2026-05-07", "settled_1d": True},
                {"signal_date": "2026-05-07", "settled_1d": False},
                {"signal_date": "2026-05-06", "settled_1d": True},
            ]
        )
        market_regime_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-08",
                    "market_price_date": "2026-05-07",
                    "market_lag_days": 1,
                }
            ]
        )

        report = build_data_freshness_report(followup_df, market_regime_df, min_settlement_ratio=1.0)

        self.assertEqual(report["expected_market_date"], "2026-05-08")
        self.assertEqual(report["settlement_date"], "2026-05-07")
        self.assertEqual(report["settlement_row_count"], 2)
        self.assertEqual(report["settled_1d_row_count"], 1)
        self.assertAlmostEqual(float(report["settled_1d_ratio"]), 0.5)
        self.assertFalse(bool(report["is_fresh"]))
        self.assertEqual(report["status"], "stale")
        self.assertIn("1日收益", str(report["reason"]))

    @patch("src.freshness.previous_a_share_trading_day", return_value=pd.Timestamp("2026-05-06"))
    @patch("src.freshness.latest_expected_market_date", return_value=pd.Timestamp("2026-05-07"))
    def test_build_data_freshness_report_marks_complete_data_fresh(self, _mock_latest_expected, _mock_previous_day) -> None:
        followup_df = pd.DataFrame(
            [
                {"signal_date": "2026-05-06", "settled_1d": True},
                {"signal_date": "2026-05-06", "settled_1d": True},
            ]
        )
        market_regime_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-07",
                    "market_price_date": "2026-05-07",
                    "market_lag_days": 0,
                }
            ]
        )

        report = build_data_freshness_report(followup_df, market_regime_df)

        self.assertTrue(bool(report["is_fresh"]))
        self.assertEqual(report["status"], "fresh")
        self.assertEqual(report["settlement_row_count"], 2)
        self.assertEqual(report["settled_1d_row_count"], 2)
        self.assertEqual(report["market_lag_days"], 0)


if __name__ == "__main__":
    unittest.main()
