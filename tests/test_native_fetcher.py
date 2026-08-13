from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src import native_fetcher


class NativeFetcherWarmCacheTest(unittest.TestCase):
    @patch("src.native_fetcher.write_csv")
    @patch("src.native_fetcher.read_csv_safely", return_value=pd.DataFrame())
    @patch("src.native_fetcher.time.sleep")
    def test_fetch_stock_price_passes_timeout_to_each_remote_source(
        self,
        _mock_sleep,
        _mock_read_csv_safely,
        _mock_write_csv,
    ) -> None:
        tx_fetch = Mock(side_effect=RuntimeError("Tencent unavailable"))
        fallback_fetch = Mock(
            return_value=pd.DataFrame(
                [{"日期": "2026-08-13", "开盘": 10, "收盘": 11, "最高": 12, "最低": 9, "成交量": 100}]
            )
        )
        fake_akshare = SimpleNamespace(stock_zh_a_hist_tx=tx_fetch, stock_zh_a_hist=fallback_fetch)

        with patch.dict(sys.modules, {"akshare": fake_akshare}):
            _prices, source = native_fetcher.fetch_stock_price("600001", force_refresh=True)

        self.assertEqual(source, "remote")
        self.assertEqual(tx_fetch.call_count, 3)
        self.assertEqual(tx_fetch.call_args.kwargs["timeout"], native_fetcher.STOCK_PRICE_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(fallback_fetch.call_args.kwargs["timeout"], native_fetcher.STOCK_PRICE_REQUEST_TIMEOUT_SECONDS)

    @patch("src.native_fetcher.fetch_stock_price")
    def test_warm_stock_price_cache_deduplicates_and_counts_sources(self, mock_fetch_stock_price) -> None:
        def fake_fetch(code: str, force_refresh: bool = False):
            mapping = {
                "000001": "cache",
                "000002": "remote",
                "000003": "missing:test",
            }
            return pd.DataFrame(), mapping[code]

        mock_fetch_stock_price.side_effect = fake_fetch

        stats = native_fetcher.warm_stock_price_cache(
            ["000001", "000002", "000001", "000003"],
            force_refresh=False,
        )

        self.assertEqual(stats["requested"], 3)
        self.assertEqual(stats["cache"], 1)
        self.assertEqual(stats["remote"], 1)
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(mock_fetch_stock_price.call_count, 3)

    @patch("src.native_fetcher.fetch_stock_price")
    def test_warm_stock_price_cache_can_force_sequential_refresh(self, mock_fetch_stock_price) -> None:
        mock_fetch_stock_price.return_value = pd.DataFrame(), "remote"

        stats = native_fetcher.warm_stock_price_cache(
            ["000001", "000002"],
            force_refresh=True,
            max_workers=1,
        )

        self.assertEqual(stats["requested"], 2)
        self.assertEqual(stats["remote"], 2)
        self.assertEqual(mock_fetch_stock_price.call_count, 2)
        self.assertTrue(all(call.kwargs["force_refresh"] for call in mock_fetch_stock_price.call_args_list))


class NativeFetcherPopularityFallbackTest(unittest.TestCase):
    @patch("src.native_fetcher.save_popularity_snapshot")
    @patch("src.native_fetcher.fetch_popularity_top100")
    @patch("src.native_fetcher.read_csv_safely")
    def test_light_capture_uses_latest_snapshot_when_popularity_fetch_times_out(
        self,
        mock_read_csv_safely,
        mock_fetch_popularity_top100,
        mock_save_popularity_snapshot,
    ) -> None:
        cached_snapshot = pd.DataFrame(
            [
                {
                    "signal_date": "2026-06-24",
                    "rank": 1,
                    "code": "000001",
                    "name": "A",
                    "popularity_score": 99,
                    "source": "10jqka",
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-06-24 09:50:00",
                },
                {
                    "signal_date": "2026-06-24",
                    "rank": 2,
                    "code": "000002",
                    "name": "B",
                    "popularity_score": 98,
                    "source": "10jqka",
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-06-24 09:50:00",
                },
            ]
        )
        mock_read_csv_safely.return_value = cached_snapshot
        mock_fetch_popularity_top100.side_effect = native_fetcher.requests.exceptions.ReadTimeout("10jqka timeout")

        result = native_fetcher.run_native_fetch(
            signal_date="2026-06-25",
            capture_type="intraday_0950",
            snapshot_time="2026-06-25 09:50:00",
            top_n=2,
            refresh_prices=False,
        )

        self.assertEqual(result["status"], "stale_popularity_cache")
        self.assertEqual(result["source"], "local_cache")
        self.assertEqual(result["fallback_signal_date"], "2026-06-24")
        self.assertEqual(result["fallback_snapshot_time"], "2026-06-24 09:50:00")
        self.assertEqual(result["popularity_rows"], 2)
        self.assertEqual(result["fetch_error_type"], "ReadTimeout")
        mock_save_popularity_snapshot.assert_not_called()

    @patch("src.native_fetcher.fetch_popularity_top100")
    @patch("src.native_fetcher.read_csv_safely", return_value=pd.DataFrame())
    def test_post_close_still_fails_when_popularity_fetch_times_out(
        self,
        _mock_read_csv_safely,
        mock_fetch_popularity_top100,
    ) -> None:
        mock_fetch_popularity_top100.side_effect = native_fetcher.requests.exceptions.ReadTimeout("10jqka timeout")

        with self.assertRaises(native_fetcher.requests.exceptions.ReadTimeout):
            native_fetcher.run_native_fetch(
                signal_date="2026-06-25",
                capture_type="post_close",
                snapshot_time="2026-06-25 15:00:00",
                top_n=2,
                refresh_prices=False,
            )


if __name__ == "__main__":
    unittest.main()
