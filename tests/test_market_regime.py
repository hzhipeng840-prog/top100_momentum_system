from __future__ import annotations

import unittest

import pandas as pd

from src.market_regime import (
    INDEX_SPECS,
    _cache_reaches_date,
    _latest_index_date,
    _merge_spot_row,
    _normalize_index_spot_df,
)


class MarketRegimeHelpersTest(unittest.TestCase):
    def test_normalize_index_spot_df_maps_snapshot_columns(self) -> None:
        raw_df = pd.DataFrame(
            [
                {
                    "\u4ee3\u7801": "sh000001",
                    "\u540d\u79f0": "\u4e0a\u8bc1\u6307\u6570",
                    "\u6700\u65b0\u4ef7": "4086.34",
                    "\u4eca\u5f00": "4074.81",
                    "\u6700\u9ad8": "4092.83",
                    "\u6700\u4f4e": "4071.08",
                    "\u6210\u4ea4\u91cf": "589950867",
                    "\u6210\u4ea4\u989d": "1136051921710",
                }
            ]
        )

        normalized = _normalize_index_spot_df(raw_df, pd.Timestamp("2026-04-27"))

        self.assertEqual(normalized.iloc[0]["symbol"], "sh000001")
        self.assertEqual(normalized.iloc[0]["date"], pd.Timestamp("2026-04-27"))
        self.assertAlmostEqual(float(normalized.iloc[0]["close"]), 4086.34, places=2)
        self.assertAlmostEqual(float(normalized.iloc[0]["open"]), 4074.81, places=2)

    def test_merge_spot_row_appends_target_day_when_daily_history_lags(self) -> None:
        stale_daily_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-04-23"),
                    "open": 4060.0,
                    "close": 4070.0,
                    "high": 4075.0,
                    "low": 4050.0,
                    "volume": 100.0,
                    "amount": 1000.0,
                },
                {
                    "date": pd.Timestamp("2026-04-24"),
                    "open": 4070.0,
                    "close": 4079.9,
                    "high": 4085.0,
                    "low": 4068.0,
                    "volume": 120.0,
                    "amount": 1200.0,
                },
            ]
        )
        spot_df = _normalize_index_spot_df(
            pd.DataFrame(
                [
                    {
                        "\u4ee3\u7801": "sh000001",
                        "\u6700\u65b0\u4ef7": "4086.3442",
                        "\u4eca\u5f00": "4074.8109",
                        "\u6700\u9ad8": "4092.8305",
                        "\u6700\u4f4e": "4071.0803",
                        "\u6210\u4ea4\u91cf": "589950867",
                        "\u6210\u4ea4\u989d": "1136051921710",
                    }
                ]
            ),
            pd.Timestamp("2026-04-27"),
        )

        merged = _merge_spot_row(stale_daily_df, spot_df, INDEX_SPECS[0])

        self.assertEqual(_latest_index_date(merged), pd.Timestamp("2026-04-27"))
        self.assertTrue(_cache_reaches_date(merged, pd.Timestamp("2026-04-27")))
        self.assertAlmostEqual(float(merged.iloc[-1]["close"]), 4086.3442, places=4)


if __name__ == "__main__":
    unittest.main()
