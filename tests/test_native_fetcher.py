from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import native_fetcher


class NativeFetcherWarmCacheTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
