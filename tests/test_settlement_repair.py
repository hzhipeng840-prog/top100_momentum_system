from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import settlement_repair


class SettlementRepairTest(unittest.TestCase):
    @patch("src.settlement_repair.warm_stock_price_cache")
    @patch("src.settlement_repair._missing_price_codes")
    def test_repair_price_caches_retries_until_target_date_is_present(
        self,
        mock_missing_price_codes,
        mock_warm_stock_price_cache,
    ) -> None:
        mock_missing_price_codes.side_effect = [["600001"], []]
        mock_warm_stock_price_cache.return_value = {
            "requested": 1,
            "remote": 1,
            "cache": 0,
            "stale_cache": 0,
            "missing": 0,
        }

        result = settlement_repair.repair_price_caches_for_target(
            {"600001"},
            "2026-05-08",
            max_attempts=3,
            max_workers=1,
            retry_sleep_seconds=0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["repaired_codes"], ["600001"])
        mock_warm_stock_price_cache.assert_called_once_with(
            ["600001"],
            force_refresh=True,
            max_workers=1,
        )

    @patch("src.settlement_repair.build_reports", return_value={"latest_push_rows": 1})
    @patch("src.settlement_repair.save_followups")
    @patch("src.settlement_repair.build_followups")
    @patch("src.settlement_repair.read_csv_safely")
    @patch("src.settlement_repair.repair_price_caches_for_target")
    @patch("src.settlement_repair._repair_targets")
    @patch("src.settlement_repair.available_strategy_versions", return_value=["v1"])
    @patch(
        "src.settlement_repair.load_settings",
        return_value={
            "default_strategy_version": "v1",
            "strategy_versions": ["v1"],
            "followup_days": [1, 3, 5, 10],
            "latest_push_limit": None,
            "strong_return_threshold_pct": 15,
            "settlement_freshness_min_ratio": 0.95,
        },
    )
    def test_run_settlement_repair_rebuilds_followups_after_price_repair(
        self,
        _mock_load_settings,
        _mock_available_strategy_versions,
        mock_repair_targets,
        mock_repair_price_caches,
        mock_read_csv_safely,
        mock_build_followups,
        mock_save_followups,
        mock_build_reports,
    ) -> None:
        stale_report = {
            "status": "stale",
            "is_fresh": False,
            "expected_market_date": "2026-05-08",
            "settlement_date": "2026-05-07",
            "settlement_row_count": 1,
            "settled_1d_row_count": 0,
            "settled_1d_ratio": 0.0,
        }
        fresh_report = {
            "status": "fresh",
            "is_fresh": True,
            "expected_market_date": "2026-05-08",
            "settlement_date": "2026-05-07",
            "settlement_row_count": 1,
            "settled_1d_row_count": 1,
            "settled_1d_ratio": 1.0,
        }
        mock_repair_targets.side_effect = [
            ({"v1": stale_report}, {"v1": {"600001"}}, {"v1": "2026-05-08"}),
            ({"v1": fresh_report}, {}, {}),
        ]
        mock_repair_price_caches.return_value = {
            "target_date": "2026-05-08",
            "requested_count": 1,
            "repaired_count": 1,
            "remaining_count": 0,
            "success": True,
        }
        signal_df = pd.DataFrame([{"signal_date": "2026-05-07", "code": "600001"}])
        followup_df = pd.DataFrame([{"signal_date": "2026-05-07", "code": "600001", "settled_1d": True}])
        mock_read_csv_safely.return_value = signal_df
        mock_build_followups.return_value = followup_df

        result = settlement_repair.run_settlement_repair(retry_sleep_seconds=0)

        self.assertTrue(result["success"])
        self.assertEqual(result["repair_code_count"], 1)
        mock_repair_price_caches.assert_called_once()
        mock_build_followups.assert_called_once()
        mock_save_followups.assert_called_once_with(followup_df, strategy_version="v1")
        mock_build_reports.assert_called_once()
        self.assertTrue(mock_build_reports.call_args.kwargs["light_mode"])


if __name__ == "__main__":
    unittest.main()
