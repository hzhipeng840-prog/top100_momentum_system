from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from src.run_modes import default_morning_snapshot_time, default_tail_snapshot_time, resolve_pipeline_mode_config, run_named_mode


class RunModesTest(unittest.TestCase):
    def test_default_tail_snapshot_time_respects_timezone(self) -> None:
        run_time = datetime.fromisoformat("2026-05-06T15:15:00+09:00")
        self.assertEqual(default_tail_snapshot_time(run_time=run_time, timezone="Asia/Shanghai"), "2026-05-06 14:30:00")

    def test_resolve_pipeline_mode_config_for_tail_capture(self) -> None:
        config = resolve_pipeline_mode_config("tail_capture", snapshot_time="2026-05-06 14:30:00")
        self.assertEqual(config.capture_type, "intraday_1430")
        self.assertTrue(config.native_fetch)
        self.assertEqual(config.snapshot_time, "2026-05-06 14:30:00")

    def test_default_morning_snapshot_time_respects_timezone(self) -> None:
        run_time = datetime.fromisoformat("2026-05-06T10:05:00+09:00")
        self.assertEqual(default_morning_snapshot_time(run_time=run_time, timezone="Asia/Shanghai"), "2026-05-06 09:50:00")

    def test_resolve_pipeline_mode_config_for_morning_capture(self) -> None:
        config = resolve_pipeline_mode_config("morning_capture", snapshot_time="2026-05-06 09:50:00")
        self.assertEqual(config.capture_type, "intraday_0950")
        self.assertTrue(config.native_fetch)
        self.assertEqual(config.snapshot_time, "2026-05-06 09:50:00")

    @patch("src.run_modes.run_settlement_repair", return_value={"mode": "settlement_repair", "success": True})
    def test_run_named_mode_settlement_repair_delegates_to_repair_service(self, mock_run_settlement_repair) -> None:
        result = run_named_mode("settlement_repair")

        self.assertTrue(result["success"])
        mock_run_settlement_repair.assert_called_once_with()

    def test_run_named_mode_nightly_reports_runs_full_local_reports_when_initial_fresh(self) -> None:
        with patch("src.run_modes.run_settlement_repair") as mock_repair:
            with patch(
                "src.run_modes.run_pipeline",
                side_effect=[
                    {"reports": {"report_mode": "light"}, "freshness": {"is_fresh": True}},
                    {"reports": {"report_mode": "full"}, "freshness": {"is_fresh": True}},
                ],
            ) as mock_run_pipeline:
                result = run_named_mode("nightly_reports")

        self.assertEqual(result["mode"], "nightly_reports")
        self.assertTrue(result["success"])
        self.assertTrue(result["nightly_settlement_repair"]["skipped"])
        mock_repair.assert_not_called()
        self.assertEqual(mock_run_pipeline.call_count, 2)
        self.assertTrue(mock_run_pipeline.call_args_list[0].kwargs["light_reports"])
        self.assertFalse(mock_run_pipeline.call_args_list[1].kwargs["light_reports"])

    def test_run_named_mode_nightly_reports_materializes_then_repairs_stale_settlement(self) -> None:
        with patch("src.run_modes.run_settlement_repair", return_value={"mode": "settlement_repair", "success": True}) as mock_repair:
            with patch(
                "src.run_modes.run_pipeline",
                side_effect=[
                    {"reports": {"report_mode": "light"}, "freshness": {"is_fresh": False}},
                    {"reports": {"report_mode": "full"}, "freshness": {"is_fresh": True}},
                ],
            ) as mock_run_pipeline:
                result = run_named_mode("nightly_reports")

        self.assertEqual(result["mode"], "nightly_reports")
        self.assertTrue(result["success"])
        self.assertEqual(result["nightly_settlement_repair"]["mode"], "settlement_repair")
        mock_repair.assert_called_once_with()
        self.assertEqual(mock_run_pipeline.call_count, 2)
        mock_run_pipeline.assert_any_call(
            native_fetch=False,
            capture_type="post_close",
            snapshot_time=None,
            force_refresh_prices=False,
            light_reports=True,
        )
        mock_run_pipeline.assert_any_call(
            native_fetch=False,
            capture_type="post_close",
            snapshot_time=None,
            force_refresh_prices=False,
            light_reports=False,
        )

    def test_run_named_mode_nightly_reports_fails_when_repair_stays_stale(self) -> None:
        with patch("src.run_modes.run_settlement_repair", return_value={"mode": "settlement_repair", "success": False}):
            with patch(
                "src.run_modes.run_pipeline",
                return_value={"reports": {"report_mode": "light"}, "freshness": {"is_fresh": False}},
            ) as mock_run_pipeline:
                result = run_named_mode("nightly_reports")

        self.assertFalse(result["success"])
        self.assertEqual(mock_run_pipeline.call_count, 1)
        self.assertTrue(mock_run_pipeline.call_args.kwargs["light_reports"])

    @patch("src.run_modes.run_backtest_service")
    @patch("src.run_modes.available_strategy_versions", return_value=["v1", "v2"])
    @patch("src.run_modes.load_settings", return_value={"strategy_versions": ["v1", "v2"]})
    def test_run_named_mode_backtest_fans_out_versions(
        self,
        _mock_load_settings,
        _mock_available_versions,
        mock_run_backtest_service,
    ) -> None:
        mock_run_backtest_service.side_effect = [
            type("Result", (), {"generated_at": "2026-05-06T09:00:00", "summary_df": [1], "rule_evaluation_df": [1]})(),
            type("Result", (), {"generated_at": "2026-05-06T09:00:01", "summary_df": [1, 2], "rule_evaluation_df": [1]})(),
        ]

        result = run_named_mode("backtest")

        self.assertEqual(result["mode"], "backtest")
        self.assertEqual(result["strategy_versions"], ["v1", "v2"])
        self.assertEqual(result["backtests"]["v1"]["summary_rows"], 1)
        self.assertEqual(result["backtests"]["v2"]["summary_rows"], 2)


if __name__ == "__main__":
    unittest.main()
