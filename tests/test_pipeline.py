from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import pipeline


class PipelineFollowupRefreshTest(unittest.TestCase):
    def test_merge_feature_history_prefers_latest_rows_and_keeps_older_dates(self) -> None:
        existing = pd.DataFrame(
            [
                {"signal_date": "2026-05-05", "code": "600001", "rank": 3, "snapshot_time": "2026-05-05 15:00:00", "close": 10},
                {"signal_date": "2026-05-06", "code": "600001", "rank": 4, "snapshot_time": "2026-05-06 14:30:00", "close": 11},
            ]
        )
        fresh = pd.DataFrame(
            [
                {"signal_date": "2026-05-06", "code": "600001", "rank": 1, "snapshot_time": "2026-05-06 15:00:00", "close": 12},
            ]
        )

        merged = pipeline._merge_feature_history(existing, fresh)

        self.assertEqual(merged["signal_date"].nunique(), 2)
        latest_row = merged[merged["signal_date"].eq("2026-05-06")].iloc[0]
        self.assertEqual(str(latest_row["snapshot_time"]), "2026-05-06 15:00:00")
        self.assertEqual(int(latest_row["rank"]), 1)

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
    @patch("src.pipeline._latest_signal_slice", return_value=pd.DataFrame([{"signal_date": "2026-04-27", "snapshot_time": "2026-04-27 15:00:00"}]))
    @patch("src.pipeline._intraday_refresh_codes", return_value=["600400", "600500"])
    @patch("src.pipeline.warm_intraday_cache")
    @patch("src.pipeline.build_data_freshness_report", return_value={"is_fresh": True, "status": "fresh", "summary": "", "reason": ""})
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
        _mock_build_data_freshness_report,
        mock_warm_intraday_cache,
        _mock_intraday_refresh_codes,
        _mock_latest_signal_slice,
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
        self.assertEqual(result["strategy_versions"], ["v1"])
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
    @patch("src.pipeline.build_latest_daily_features", return_value=pd.DataFrame())
    @patch("src.pipeline.read_csv_safely", return_value=pd.DataFrame())
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
        _mock_read_csv_safely,
        _mock_build_latest_daily_features,
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

        _mock_run_native_fetch.assert_called_once_with(
            capture_type="intraday_1430",
            snapshot_time=None,
            top_n=100,
            refresh_prices=False,
            force_refresh_prices=False,
        )
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
        _mock_warm_stock_price_cache.assert_not_called()
        _mock_build_followups.assert_not_called()
        _mock_save_followups.assert_not_called()

    @patch("src.pipeline.read_csv_safely", return_value=pd.DataFrame())
    @patch("src.pipeline._followup_refresh_codes", return_value={"600001"})
    @patch("src.pipeline.build_reports", return_value={})
    @patch("src.pipeline.save_followups")
    @patch("src.pipeline.build_followups")
    @patch("src.pipeline.warm_stock_price_cache")
    @patch("src.pipeline.save_signals")
    @patch("src.pipeline.build_signals")
    @patch("src.pipeline.save_market_regime")
    @patch("src.pipeline.build_market_regime", return_value=pd.DataFrame([{"signal_date": "2026-04-27", "market_regime": "strong", "market_1d_pct": 0.1, "market_5d_pct": 0.2, "market_lag_days": 0, "market_price_date": "2026-04-27"}]))
    @patch("src.pipeline.save_daily_features")
    @patch("src.pipeline.build_daily_features", return_value=pd.DataFrame([{"signal_date": "2026-04-27", "code": "600001"}]))
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
    def test_run_pipeline_warms_stock_cache_before_building_followups(
        self,
        _mock_load_settings,
        _mock_should_skip_market_fetch,
        _mock_run_native_fetch,
        _mock_build_daily_features,
        _mock_save_daily_features,
        _mock_build_market_regime,
        _mock_save_market_regime,
        mock_build_signals,
        _mock_save_signals,
        mock_warm_stock_price_cache,
        mock_build_followups,
        _mock_save_followups,
        _mock_build_reports,
        mock_followup_refresh_codes,
        _mock_read_csv_safely,
    ) -> None:
        events: list[str] = []

        mock_build_signals.return_value = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-27",
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-27 15:00:00",
                    "code": "600001",
                    "is_pushed": True,
                    "emotion_score": 92,
                    "rank": 2,
                }
            ]
        )

        def warm_side_effect(codes, **kwargs):
            events.append(f"warm:{','.join(codes)}")
            return {"requested": len(codes), "remote": len(codes), "cache": 0, "stale_cache": 0, "missing": 0}

        def build_followups_side_effect(signal_df, days, strategy_version):
            events.append(f"build:{strategy_version}")
            return pd.DataFrame(
                [
                    {
                        "strategy_version": strategy_version,
                        "signal_date": "2026-04-27",
                        "code": "600001",
                        "observed_days": 1,
                        "settled_1d": True,
                    }
                ]
            )

        mock_warm_stock_price_cache.side_effect = warm_side_effect
        mock_build_followups.side_effect = build_followups_side_effect

        result = pipeline.run_pipeline(native_fetch=True, capture_type="post_close")

        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(events[0].startswith("warm:"))
        self.assertTrue(events[1].startswith("build:"))
        self.assertEqual(result["followup_price_cache"]["requested"], 1)
        _mock_build_reports.assert_called_once()
        self.assertFalse(_mock_build_reports.call_args.kwargs["light_mode"])

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
    @patch("src.pipeline.build_latest_daily_features", return_value=pd.DataFrame())
    @patch("src.pipeline.read_csv_safely", return_value=pd.DataFrame())
    @patch(
        "src.pipeline.load_popularity",
        return_value=pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-28",
                    "rank": 1,
                    "code": "600001",
                    "name": "娴嬭瘯鑲",
                    "popularity_score": 100,
                    "source": "10jqka",
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-04-28 09:50:00",
                },
                {
                    "signal_date": "2026-04-28",
                    "rank": 2,
                    "code": "600002",
                    "name": "娴嬭瘯鑲",
                    "popularity_score": 90,
                    "source": "10jqka",
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-04-28 09:50:00",
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
    def test_run_pipeline_warms_intraday_feature_cache_for_intraday_v2(
        self,
        _mock_load_settings,
        _mock_should_skip_market_fetch,
        _mock_run_native_fetch,
        _mock_load_popularity,
        _mock_read_csv_safely,
        _mock_build_latest_daily_features,
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

        result = pipeline.run_pipeline(native_fetch=True, capture_type="intraday_0950")

        _mock_run_native_fetch.assert_called_once_with(
            capture_type="intraday_0950",
            snapshot_time=None,
            top_n=100,
            refresh_prices=False,
            force_refresh_prices=False,
        )
        mock_warm_intraday_cache.assert_called_once_with(
            ["600001", "600002"],
            trade_date="2026-04-28",
            capture_type="intraday_0950",
            snapshot_time="2026-04-28 09:50:00",
            refresh_snapshot=True,
            refresh_bars=False,
            force_refresh_snapshot=True,
            force_refresh_bars=False,
        )
        _mock_warm_stock_price_cache.assert_not_called()
        _mock_build_followups.assert_not_called()
        _mock_save_followups.assert_not_called()
        _mock_build_reports.assert_called_once()
        self.assertTrue(_mock_build_reports.call_args.kwargs["light_mode"])
        self.assertEqual(result["strategy_versions"], ["v2"])
        self.assertEqual(result["intraday_feature_cache"], {"requested": 2})

    @patch("src.pipeline.build_reports", return_value={})
    @patch("src.pipeline.save_followups")
    @patch("src.pipeline.build_followups", return_value=pd.DataFrame())
    @patch("src.pipeline.save_signals")
    @patch("src.pipeline.build_signals", return_value=pd.DataFrame())
    @patch("src.pipeline.save_market_regime")
    @patch("src.pipeline.build_market_regime", return_value=pd.DataFrame())
    @patch("src.pipeline.save_daily_features")
    @patch("src.pipeline.build_daily_features", return_value=pd.DataFrame([{"signal_date": "2026-05-06", "code": "600001", "rank": 1, "snapshot_time": "2026-05-06 15:00:00"}]))
    @patch("src.pipeline.load_popularity", return_value=pd.DataFrame([{"signal_date": "2026-05-06", "rank": 1, "code": "600001"}]))
    @patch("src.pipeline.read_csv_safely")
    @patch(
        "src.pipeline.load_settings",
        return_value={
            "strategy_versions": ["v1", "v2", "v3"],
            "default_strategy_version": "v1",
            "default_capture_type": "post_close",
            "signal_min_score": 60,
            "latest_push_limit": None,
            "strong_return_threshold_pct": 15,
            "followup_days": [1, 3, 5, 10],
            "refresh_price_cache": False,
            "refresh_market_cache": False,
            "refresh_intraday_cache": False,
        },
    )
    def test_run_pipeline_reuses_existing_feature_history_when_popularity_is_sparse(
        self,
        _mock_load_settings,
        _mock_read_csv_safely,
        _mock_load_popularity,
        _mock_build_daily_features,
        mock_save_daily_features,
        _mock_build_market_regime,
        _mock_save_market_regime,
        mock_build_signals,
        _mock_save_signals,
        _mock_build_followups,
        _mock_save_followups,
        _mock_build_reports,
    ) -> None:
        existing_feature_history = pd.DataFrame(
            [
                {"signal_date": "2026-05-05", "code": "600001", "rank": 2, "snapshot_time": "2026-05-05 15:00:00"},
                {"signal_date": "2026-05-06", "code": "600002", "rank": 3, "snapshot_time": "2026-05-06 15:00:00"},
            ]
        )

        def fake_read(path):
            return existing_feature_history if str(path).endswith("daily_features.csv") else pd.DataFrame()

        _mock_read_csv_safely.side_effect = fake_read

        pipeline.run_pipeline(native_fetch=False)

        saved_feature_df = mock_save_daily_features.call_args.args[0]
        self.assertEqual(saved_feature_df["signal_date"].nunique(), 2)
        first_signal_features = mock_build_signals.call_args_list[0].args[0]
        third_signal_features = mock_build_signals.call_args_list[2].args[0]
        self.assertEqual(first_signal_features["signal_date"].nunique(), 1)
        self.assertEqual(third_signal_features["signal_date"].nunique(), 1)

    @patch("src.pipeline.warm_intraday_cache")
    @patch("src.pipeline.build_reports", return_value={})
    @patch("src.pipeline.save_followups")
    @patch("src.pipeline.build_followups")
    @patch("src.pipeline.warm_stock_price_cache", return_value={})
    @patch("src.pipeline.save_signals")
    @patch("src.pipeline.build_signals")
    @patch("src.pipeline.save_market_regime")
    @patch("src.pipeline.build_market_regime")
    @patch("src.pipeline.save_daily_features")
    @patch("src.pipeline.build_daily_features")
    @patch("src.pipeline.run_native_fetch", return_value={"status": "ok"})
    @patch("src.pipeline.should_skip_market_fetch", return_value={"skip": False, "skip_reason_code": "", "reason": "", "expected_signal_date": "2026-05-08"})
    @patch("src.pipeline.read_csv_safely", return_value=pd.DataFrame())
    @patch("src.pipeline.load_settings", return_value={"top_n": 100, "default_capture_type": "post_close", "strategy_versions": ["v1"], "followup_days": [1, 3, 5, 10], "refresh_price_cache": True, "refresh_market_cache": True, "refresh_intraday_cache": True, "intraday_cache_push_only": True, "intraday_cache_limit": 20, "settlement_freshness_min_ratio": 1.0})
    @patch("src.freshness.previous_a_share_trading_day", return_value=pd.Timestamp("2026-05-07"))
    @patch("src.freshness.latest_expected_market_date", return_value=pd.Timestamp("2026-05-08"))
    def test_run_pipeline_marks_full_run_stale_when_settlement_is_unfinished(
        self,
        _mock_latest_expected_market_date,
        _mock_previous_trading_day,
        _mock_load_settings,
        _mock_read_csv_safely,
        _mock_should_skip_market_fetch,
        _mock_run_native_fetch,
        mock_build_daily_features,
        _mock_save_daily_features,
        mock_build_market_regime,
        _mock_save_market_regime,
        mock_build_signals,
        _mock_save_signals,
        _mock_warm_stock_price_cache,
        mock_build_followups,
        _mock_save_followups,
        _mock_build_reports,
        mock_warm_intraday_cache,
    ) -> None:
        mock_build_daily_features.return_value = pd.DataFrame([{"signal_date": "2026-05-08", "code": "600001", "rank": 1}])
        mock_build_market_regime.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-08", "market_price_date": "2026-05-07", "market_lag_days": 1}]
        )
        mock_build_signals.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-08", "code": "600001", "is_pushed": True, "rank": 1, "capture_type": "post_close"}]
        )
        mock_build_followups.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-07", "code": "600001", "settled_1d": False, "observed_days": 0}]
        )

        result = pipeline.run_pipeline(native_fetch=True, capture_type="post_close")

        self.assertEqual(result["data"]["status"], "stale_settlement")
        self.assertEqual(result["data"]["freshness_status"], "stale")
        self.assertFalse(bool(result["freshness"]["is_fresh"]))
        mock_warm_intraday_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
