from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from src.paths import PROJECT_ROOT
from src.workflow_tasks import (
    build_tail_snapshot_time,
    build_workflow_command,
    run_workflow_mode,
    validate_workflow_mode,
    write_workflow_summary,
)


class WorkflowTasksTest(unittest.TestCase):
    def _workspace_tempdir(self) -> Path:
        tmp_root = PROJECT_ROOT / "workflow_artifacts" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        path = tmp_root / f"case_{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_validate_workflow_mode_normalizes_case(self) -> None:
        self.assertEqual(validate_workflow_mode(" FULL "), "full")

    def test_build_tail_snapshot_time_uses_requested_timezone(self) -> None:
        run_time = datetime.fromisoformat("2026-05-06T07:15:00+09:00")
        self.assertEqual(
            build_tail_snapshot_time(run_time=run_time, timezone="Asia/Shanghai"),
            "2026-05-06 14:30:00",
        )

    def test_build_workflow_command_for_tail_capture_is_explicit(self) -> None:
        command = build_workflow_command(
            "tail_capture",
            snapshot_time="2026-05-06 14:30:00",
            python_executable="python",
        )
        self.assertEqual(
            command,
            [
                "python",
                "daily_job.py",
                "--capture-type",
                "intraday_1430",
                "--snapshot-time",
                "2026-05-06 14:30:00",
            ],
        )

    def test_build_workflow_command_for_recompute(self) -> None:
        command = build_workflow_command("recompute", python_executable="python")
        self.assertEqual(command, ["python", "daily_job.py", "--no-fetch"])

    def test_write_workflow_summary_includes_pipeline_status(self) -> None:
        tmpdir = self._workspace_tempdir()
        try:
            summary_path = tmpdir / "summary.md"
            write_workflow_summary(
                summary_path,
                {
                    "success": True,
                    "mode": "full",
                    "started_at": "2026-05-06T17:10:00+08:00",
                    "finished_at": "2026-05-06T17:10:05+08:00",
                    "returncode": 0,
                    "command_display": "python daily_job.py --capture-type post_close",
                    "log_path": "workflow_artifacts/logs/full.log",
                    "payload": {
                        "strategy_versions": ["v1", "v2", "v3"],
                        "default_strategy_version": "v1",
                        "data": {"status": "ok"},
                        "features": {"rows": 1200},
                        "signals": {"rows": 100, "pushed_rows": 14},
                        "followups": {"rows": 900},
                    },
                },
            )

            text = summary_path.read_text(encoding="utf-8")
            self.assertIn("Data Status: `ok`", text)
            self.assertIn("Strategy Versions: `v1, v2, v3`", text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @patch("src.workflow_tasks.subprocess.run")
    def test_run_workflow_mode_writes_log_and_summary(self, mock_run) -> None:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"data":{"status":"ok"},"strategy_versions":["v1"],"default_strategy_version":"v1","features":{"rows":1},"signals":{"rows":1,"pushed_rows":1},"followups":{"rows":1}}'
        mock_run.return_value.stderr = ""

        tmpdir = self._workspace_tempdir()
        try:
            summary_path = tmpdir / "workflow_summary.md"
            log_dir = tmpdir / "logs"
            result = run_workflow_mode(
                "tail_capture",
                snapshot_time="2026-05-06 14:30:00",
                summary_path=summary_path,
                log_dir=log_dir,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["snapshot_time"], "2026-05-06 14:30:00")
            self.assertTrue(summary_path.exists())
            self.assertEqual(len(list(log_dir.glob("tail_capture_*.log"))), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
