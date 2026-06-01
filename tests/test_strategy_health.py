from __future__ import annotations

import unittest

import pandas as pd

from src.strategy_health import apply_v3_health_cooldown


class StrategyHealthCooldownTest(unittest.TestCase):
    def test_v3_health_cools_high_score_push_only(self) -> None:
        history_signals = [
            {
                "signal_date": f"2026-05-{day:02d}",
                "code": f"0000{day:02d}",
                "name": f"样本{day}",
                "emotion_score": 90.0,
                "push_level": "重点观察",
                "is_pushed": True,
            }
            for day in range(1, 12)
        ]
        current_signals = [
            {
                "signal_date": "2026-05-15",
                "code": "300001",
                "name": "高分强推",
                "emotion_score": 106.0,
                "push_level": "强推观察",
                "is_pushed": True,
                "risks": "-",
                "suggested_action": "原执行建议",
            },
            {
                "signal_date": "2026-05-15",
                "code": "300002",
                "name": "普通推送",
                "emotion_score": 88.0,
                "push_level": "重点观察",
                "is_pushed": True,
                "risks": "-",
                "suggested_action": "原执行建议",
            },
        ]
        signal_df = pd.DataFrame(history_signals + current_signals)
        followup_df = pd.DataFrame(
            [
                {
                    "signal_date": f"2026-05-{day:02d}",
                    "code": f"0000{day:02d}",
                    "tail_next_close_pct": -2.0,
                    "settled_tail_next_day": True,
                    "latest_price_date": f"2026-05-{day + 1:02d}",
                }
                for day in range(1, 12)
            ]
        )

        result = apply_v3_health_cooldown(signal_df, followup_df, strategy_version="v3")

        high_score = result[result["code"].eq("300001")].iloc[0]
        normal_score = result[result["code"].eq("300002")].iloc[0]
        self.assertFalse(bool(high_score["is_pushed"]))
        self.assertEqual(str(high_score["push_level"]), "观察池")
        self.assertTrue(bool(high_score["v3_health_cooldown"]))
        self.assertIn("v3健康度冷却", str(high_score["risks"]))
        self.assertTrue(bool(normal_score["is_pushed"]))
        self.assertEqual(str(normal_score["push_level"]), "重点观察")

    def test_non_v3_strategy_is_unchanged(self) -> None:
        signal_df = pd.DataFrame(
            [{"signal_date": "2026-05-15", "code": "300001", "emotion_score": 106.0, "is_pushed": True}]
        )
        result = apply_v3_health_cooldown(signal_df, pd.DataFrame(), strategy_version="v1")

        self.assertEqual(result.to_dict("records"), signal_df.to_dict("records"))


if __name__ == "__main__":
    unittest.main()
