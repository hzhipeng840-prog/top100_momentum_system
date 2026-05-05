from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.intraday_fetcher import warm_intraday_cache


class IntradayFetcherWarmCacheTest(unittest.TestCase):
    @patch("src.intraday_fetcher.fetch_intraday_bars")
    @patch("src.intraday_fetcher.fetch_intraday_snapshot")
    def test_warm_intraday_cache_deduplicates_and_counts_sources(
        self,
        mock_fetch_intraday_snapshot,
        mock_fetch_intraday_bars,
    ) -> None:
        mock_fetch_intraday_snapshot.side_effect = [
            (pd.DataFrame(), "single_remote"),
            (pd.DataFrame(), "cache"),
        ]
        mock_fetch_intraday_bars.side_effect = [
            (pd.DataFrame(), "akshare.stock_zh_a_hist_min_em"),
            (pd.DataFrame(), "cache"),
        ]

        stats = warm_intraday_cache(
            ["000001", "000002", "000001"],
            trade_date="2026-04-27",
            force_refresh_snapshot=True,
            force_refresh_bars=False,
        )

        self.assertEqual(stats["requested"], 2)
        self.assertEqual(stats["snapshot"]["requested"], 2)
        self.assertEqual(stats["snapshot"]["remote"], 1)
        self.assertEqual(stats["snapshot"]["cache"], 1)
        self.assertEqual(stats["bars"]["requested"], 2)
        self.assertEqual(stats["bars"]["remote"], 1)
        self.assertEqual(stats["bars"]["cache"], 1)
        self.assertEqual(mock_fetch_intraday_snapshot.call_count, 2)
        self.assertEqual(mock_fetch_intraday_bars.call_count, 2)


if __name__ == "__main__":
    unittest.main()
