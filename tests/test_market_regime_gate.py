from __future__ import annotations

import unittest

import pandas as pd

from src.signals import build_signals, score_signal


class MarketRegimeGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_row = {
            "signal_date": "2026-05-07",
            "rank": 35,
            "day_return_pct": 4.2,
            "close_position": 0.72,
            "volume_ratio_5": 0.95,
            "pre5_return_pct": 7.0,
            "dist_ma20_pct": 5.0,
            "consecutive_days": 1,
            "appearance_count": 2,
            "rank_change": 2,
            "one_word_like": False,
            "limit_up_like": False,
            "upper_shadow_pct": 0.08,
            "price_status": "ok",
            "market_1d_pct": 0.8,
            "market_5d_pct": 1.2,
        }

    def test_market_regime_adjusts_score(self) -> None:
        neutral = score_signal(self.base_row, strategy_version="v3")
        strong = score_signal(
            dict(self.base_row, market_regime="强势", relative_1d_pct=1.8, relative_5d_pct=2.6),
            strategy_version="v3",
        )
        weak = score_signal(
            dict(self.base_row, market_regime="弱势", relative_1d_pct=-1.8, relative_5d_pct=-2.6),
            strategy_version="v3",
        )

        self.assertGreater(strong["emotion_score"], neutral["emotion_score"])
        self.assertLess(weak["emotion_score"], neutral["emotion_score"])
        self.assertIn("市场强势", strong["reasons"])
        self.assertIn("弱市", weak["risks"])

    def test_build_signals_uses_market_regime_df(self) -> None:
        feature_row = dict(self.base_row)
        feature_row.pop("market_1d_pct", None)
        feature_row.pop("market_5d_pct", None)
        feature_df = pd.DataFrame([feature_row])
        market_regime_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-07",
                    "market_regime": "强势",
                    "market_1d_pct": 0.8,
                    "market_5d_pct": 1.2,
                }
            ]
        )

        neutral = score_signal(self.base_row, strategy_version="v3")
        result = build_signals(feature_df=feature_df, market_regime_df=market_regime_df, strategy_version="v3")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["market_regime"], "强势")
        self.assertGreater(float(result.iloc[0]["emotion_score"]), neutral["emotion_score"])


if __name__ == "__main__":
    unittest.main()
