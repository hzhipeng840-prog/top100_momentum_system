from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import pipeline


class PipelineFollowupRefreshTest(unittest.TestCase):
    @patch("src.pipeline.read_csv_safely")
    def test_followup_refresh_codes_include_unsettled_non_pushed_samples(self, mock_read_csv_safely) -> None:
        history_df = pd.DataFrame([{"code": "600001"}])
        followup_df = pd.DataFrame(
            [
                {"code": "300067", "settled_10d": False, "observed_days": 2},
                {"code": "603738", "settled_10d": True, "observed_days": 10},
            ]
        )

        def fake_read(path):
            path_text = str(path)
            if path_text.endswith("fast_strategy_history.csv"):
                return history_df
            if path_text.endswith("followups.csv"):
                return followup_df
            return pd.DataFrame()

        mock_read_csv_safely.side_effect = fake_read

        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-22", "code": "300067", "is_pushed": False},
                {"signal_date": "2026-04-22", "code": "603738", "is_pushed": False},
                {"signal_date": "2026-04-22", "code": "600726", "is_pushed": True},
            ]
        )

        codes = pipeline._followup_refresh_codes(signal_df, followup_days=[1, 3, 5, 10])

        self.assertIn("300067", codes)
        self.assertIn("600001", codes)
        self.assertIn("600726", codes)
        self.assertNotIn("603738", codes)

    def test_intraday_refresh_codes_focus_latest_pushed_samples(self) -> None:
        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-26", "snapshot_time": "2026-04-26 15:00:00", "code": "600100", "is_pushed": True, "emotion_score": 80, "rank": 3},
                {"signal_date": "2026-04-27", "capture_type": "intraday_1430", "snapshot_time": "2026-04-27 14:30:00", "code": "600200", "is_pushed": True, "emotion_score": 70, "rank": 5},
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600300", "is_pushed": False, "emotion_score": 95, "rank": 1},
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600400", "is_pushed": True, "emotion_score": 88, "rank": 4},
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600500", "is_pushed": True, "emotion_score": 92, "rank": 2},
            ]
        )

        codes = pipeline._intraday_refresh_codes(signal_df, pushed_only=True, limit=5)

        self.assertEqual(codes, ["600500", "600400"])

    @patch("src.pipeline.latest_expected_market_date", return_value=pd.Timestamp("2026-04-27"))
    @patch("src.pipeline.warm_intraday_cache")
    @patch("src.pipeline.build_reports", return_value={})
    @patch("src.pipeline.save_followups")
    @patch("src.pipeline.build_followups", return_value=pd.DataFrame())
    @patch("src.pipeline.warm_stock_price_cache", return_value={})
    @patch("src.pipeline.save_signals")
    @patch("src.pipeline.build_signals")
    @patch("src.pipeline.save_market_regime")
    @patch("src.pipeline.build_market_regime", return_value=pd.DataFrame([{"signal_date": "2026-04-27", "market_regime": "normal", "market_1d_pct": 0.1, "market_5d_pct": 0.2}]))
    @patch("src.pipeline.warm_market_index_cache", return_value={})
    @patch("src.pipeline.save_daily_features")
    @patch("src.pipeline.build_daily_features", return_value=pd.DataFrame([{"signal_date": "2026-04-27", "code": "600500"}]))
    @patch("src.pipeline.run_native_fetch", return_value={"status": "ok"})
    @patch("src.pipeline.should_skip_market_fetch", return_value={"skip": False, "skip_reason_code": "", "reason": "", "expected_signal_date": "2026-04-27"})
    @patch(
        "src.pipeline.load_settings",
        return_value={
            "top_n": 100,
            "default_capture_type": "post_close",
            "signal_min_score": 60,
            "latest_push_limit": None,
            "strong_return_threshold_pct": 15,
            "followup_days": [1, 3, 5, 10],
            "refresh_price_cache": True,
            "refresh_market_cache": True,
            "refresh_intraday_cache": True,
            "intraday_cache_push_only": True,
            "intraday_cache_limit": 20,
        },
    )
    def test_run_pipeline_warms_intraday_cache_for_latest_pushed_codes(
        self,
        _mock_load_settings,
        _mock_should_skip_market_fetch,
        _mock_run_native_fetch,
        _mock_build_daily_features,
        _mock_save_daily_features,
        _mock_warm_market_index_cache,
        _mock_build_market_regime,
        _mock_save_market_regime,
        mock_build_signals,
        _mock_save_signals,
        _mock_warm_stock_price_cache,
        _mock_build_followups,
        _mock_save_followups,
        _mock_build_reports,
        mock_warm_intraday_cache,
        _mock_latest_expected_market_date,
    ) -> None:
        mock_build_signals.return_value = pd.DataFrame(
            [
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600500", "is_pushed": True, "emotion_score": 92, "rank": 2},
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600400", "is_pushed": True, "emotion_score": 88, "rank": 4},
                {"signal_date": "2026-04-27", "capture_type": "post_close", "snapshot_time": "2026-04-27 15:00:00", "code": "600300", "is_pushed": False, "emotion_score": 95, "rank": 1},
            ]
        )
        mock_warm_intraday_cache.return_value = {"requested": 2}

        result = pipeline.run_pipeline(native_fetch=True, capture_type="post_close")

        mock_warm_intraday_cache.assert_called_once_with(
            ["600400", "600500"],
            trade_date="2026-04-27",
            capture_type="post_close",
            refresh_snapshot=True,
            refresh_bars=True,
            force_refresh_snapshot=True,
            force_refresh_bars=False,
        )
        self.assertEqual(result["intraday_cache"], {"requested": 2})

    @patch("src.pipeline.warm_intraday_cache")
    @patch("src.pipeline.build_reports", return_value={})
    @patch("src.pipeline.save_followups")
    @patch("src.pipeline.build_followups", return_value=pd.DataFrame())
    @patch("src.pipeline.warm_stock_price_cache", return_value={})
    @patch("src.pipeline.save_signals")
    @patch("src.pipeline.build_signals", return_value=pd.DataFrame())
    @patch("src.pipeline.save_market_regime")
    @patch("src.pipeline.build_market_regime", return_value=pd.DataFrame())
    @patch("src.pipeline.save_daily_features")
    @patch("src.pipeline.build_daily_features", return_value=pd.DataFrame())
    @patch(
        "src.pipeline.load_popularity",
        return_value=pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "测试股A",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "intraday_1430",
                    "snapshot_time": "2026-04-28 14:30:00",
                },
                {
                    "signal_date": "2026-04-28",
                    "rank": 2,
                    "code": "600002",
                    "name": "测试股B",
                    "popularity_score": 90,
                    "source": "10jqka",
                    "capture_type": "intraday_1430",
                    "snapshot_time": "2026-04-28 14:30:00",
                },
            ]
        ),
    )
    @patch("src.pipeline.run_native_fetch", return_value={"status": "ok"})
    @patch("src.pipeline.should_skip_market_fetch", return_value={"skip": False, "skip_reason_code": "", "reason": "", "expected_signal_date": "2026-04-28"})
    @patch(
        "src.pipeline.load_settings",
        return_value={
            "top_n": 100,
            "default_capture_type": "post_close",
            "signal_min_score": 60,
            "latest_push_limit": None,
            "strong_return_threshold_pct": 15,
            "followup_days": [1, 3, 5, 10],
            "refresh_price_cache": True,
            "refresh_market_cache": True,
            "refresh_intraday_cache": True,
            "intraday_cache_push_only": True,
            "intraday_cache_limit": 20,
        },
    )
    def test_run_pipeline_warms_intraday_feature_cache_for_intraday_v3(
        self,
        _mock_load_settings,
        _mock_should_skip_market_fetch,
        _mock_run_native_fetch,
        _mock_load_popularity,
        _mock_build_daily_features,
        _mock_save_daily_features,
        _mock_build_market_regime,
        _mock_save_market_regime,
        _mock_build_signals,
        _mock_save_signals,
        _mock_warm_stock_price_cache,
        _mock_build_followups,
        _mock_save_followups,
        _mock_build_reports,
        mock_warm_intraday_cache,
    ) -> None:
        mock_warm_intraday_cache.return_value = {"requested": 2}

        result = pipeline.run_pipeline(native_fetch=True, capture_type="intraday_1430")

        mock_warm_intraday_cache.assert_called_once_with(
            ["600001", "600002"],
            trade_date="2026-04-28",
            capture_type="intraday_1430",
            snapshot_time="2026-04-28 14:30:00",
            refresh_snapshot=True,
            refresh_bars=False,
            force_refresh_snapshot=True,
            force_refresh_bars=False,
        )
        self.assertEqual(result["intraday_feature_cache"], {"requested": 2})


if __name__ == "__main__":
    unittest.main()
