from __future__ import annotations

import unittest
from datetime import datetime

from src.trading_calendar import is_within_capture_window


class TradingCalendarCaptureWindowTest(unittest.TestCase):
    def test_intraday_tail_capture_allows_short_delay_window(self) -> None:
        self.assertTrue(is_within_capture_window("intraday_1430", now=datetime.fromisoformat("2026-05-07T14:38:00+08:00")))
        self.assertFalse(is_within_capture_window("intraday_1430", now=datetime.fromisoformat("2026-05-07T20:19:00+08:00")))

    def test_post_close_window_rejects_after_midnight_delay(self) -> None:
        self.assertTrue(is_within_capture_window("post_close", now=datetime.fromisoformat("2026-05-07T17:18:00+08:00")))
        self.assertFalse(is_within_capture_window("post_close", now=datetime.fromisoformat("2026-05-08T00:12:00+08:00")))


if __name__ == "__main__":
    unittest.main()
