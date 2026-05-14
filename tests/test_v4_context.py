from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src import v4_context


class V4ContextTest(unittest.TestCase):
    @patch("src.v4_context._ensure_cyq_context")
    @patch("src.v4_context._ensure_lhb_context")
    @patch("src.v4_context._ensure_fund_flow_context")
    @patch("src.v4_context._ensure_notice_context")
    def test_enrich_v4_context_merges_external_fields(
        self,
        mock_notice_context,
        mock_fund_flow_context,
        mock_lhb_context,
        mock_cyq_context,
    ) -> None:
        feature_df = pd.DataFrame(
            [
                {"signal_date": "2026-05-12", "code": "002580", "rank": 1},
                {"signal_date": "2026-05-12", "code": "600001", "rank": 2},
            ]
        )
        mock_notice_context.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-12", "code": "002580", "announcement_summary": "签订大单", "event_summary": "重大合同"}]
        )
        mock_fund_flow_context.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-12", "code": "002580", "capital_flow_pct": 3.2, "capital_flow_signal": 3.2}]
        )
        mock_lhb_context.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-12", "code": "002580", "dragon_tiger_positive": True}]
        )
        mock_cyq_context.return_value = pd.DataFrame(
            [{"signal_date": "2026-05-12", "code": "002580", "profit_ratio": 61.5, "concentration_90": 12.8}]
        )

        enriched = v4_context.enrich_v4_context(feature_df)

        first_row = enriched[enriched["code"].eq("002580")].iloc[0]
        second_row = enriched[enriched["code"].eq("600001")].iloc[0]
        self.assertEqual(first_row["announcement_summary"], "签订大单")
        self.assertEqual(first_row["event_summary"], "重大合同")
        self.assertAlmostEqual(float(first_row["capital_flow_signal"]), 3.2)
        self.assertTrue(bool(first_row["dragon_tiger_positive"]))
        self.assertAlmostEqual(float(first_row["profit_ratio"]), 61.5)
        self.assertAlmostEqual(float(first_row["concentration_90"]), 12.8)
        self.assertFalse(bool(second_row["dragon_tiger_positive"]))
        self.assertTrue(pd.isna(second_row["board_strength"]))

    def test_fetch_notice_rows_groups_titles_and_types(self) -> None:
        raw_notice_df = pd.DataFrame(
            [
                {
                    "代码": "002580",
                    "公告标题": "关于签订框架协议的公告",
                    "公告类型": "重大合同",
                    "公告日期": "2026-05-12",
                },
                {
                    "代码": "002580",
                    "公告标题": "关于签订框架协议的公告",
                    "公告类型": "重大合同",
                    "公告日期": "2026-05-12",
                },
                {
                    "代码": "002580",
                    "公告标题": "关于项目中标的公告",
                    "公告类型": "重大合同",
                    "公告日期": "2026-05-12",
                },
            ]
        )

        with patch("akshare.stock_notice_report", return_value=raw_notice_df):
            grouped = v4_context._fetch_notice_rows("2026-05-12")

        self.assertEqual(len(grouped), 1)
        row = grouped.iloc[0]
        self.assertEqual(str(row["signal_date"]), "2026-05-12")
        self.assertEqual(str(row["code"]), "002580")
        self.assertIn("关于签订框架协议的公告", str(row["announcement_summary"]))
        self.assertIn("关于项目中标的公告", str(row["announcement_summary"]))
        self.assertIn("重大合同", str(row["event_summary"]))


if __name__ == "__main__":
    unittest.main()
