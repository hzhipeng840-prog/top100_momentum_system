from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from src.github_dispatch import (
    DEFAULT_REPOSITORY,
    DEFAULT_WORKFLOW_FILE,
    WorkflowDispatchConfig,
    build_workflow_dispatch_headers,
    build_workflow_dispatch_payload,
    build_workflow_dispatch_url,
    config_from_env,
    dispatch_workflow,
)


class GithubDispatchTest(unittest.TestCase):
    def test_build_workflow_dispatch_url_uses_repo_and_file(self) -> None:
        url = build_workflow_dispatch_url("owner/repo", "pipeline.yml")
        self.assertEqual(url, "https://api.github.com/repos/owner/repo/actions/workflows/pipeline.yml/dispatches")

    def test_build_workflow_dispatch_payload_preserves_mode_and_bool(self) -> None:
        payload = build_workflow_dispatch_payload(ref="main", mode="tail_capture", force_refresh_prices=True)
        self.assertEqual(payload["ref"], "main")
        self.assertEqual(payload["inputs"]["mode"], "tail_capture")
        self.assertIs(payload["inputs"]["force_refresh_prices"], True)

    def test_build_workflow_dispatch_headers_requires_token(self) -> None:
        headers = build_workflow_dispatch_headers("token-123")
        self.assertEqual(headers["Authorization"], "Bearer token-123")
        with self.assertRaises(ValueError):
            build_workflow_dispatch_headers("")

    def test_config_from_env_reads_defaults(self) -> None:
        original_token = os.environ.get("GITHUB_TOKEN")
        original_repo = os.environ.get("GITHUB_REPOSITORY")
        original_workflow = os.environ.get("GITHUB_WORKFLOW_FILE")
        original_ref = os.environ.get("GITHUB_WORKFLOW_REF")
        try:
            os.environ["GITHUB_TOKEN"] = "abc"
            os.environ.pop("GITHUB_REPOSITORY", None)
            os.environ.pop("GITHUB_WORKFLOW_FILE", None)
            os.environ.pop("GITHUB_WORKFLOW_REF", None)
            config = config_from_env(mode="full")
            self.assertEqual(config.repository, DEFAULT_REPOSITORY)
            self.assertEqual(config.workflow_file, DEFAULT_WORKFLOW_FILE)
            self.assertEqual(config.ref, "main")
            self.assertEqual(config.token, "abc")
        finally:
            if original_token is None:
                os.environ.pop("GITHUB_TOKEN", None)
            else:
                os.environ["GITHUB_TOKEN"] = original_token
            if original_repo is None:
                os.environ.pop("GITHUB_REPOSITORY", None)
            else:
                os.environ["GITHUB_REPOSITORY"] = original_repo
            if original_workflow is None:
                os.environ.pop("GITHUB_WORKFLOW_FILE", None)
            else:
                os.environ["GITHUB_WORKFLOW_FILE"] = original_workflow
            if original_ref is None:
                os.environ.pop("GITHUB_WORKFLOW_REF", None)
            else:
                os.environ["GITHUB_WORKFLOW_REF"] = original_ref

    @patch("src.github_dispatch.requests.post")
    def test_dispatch_workflow_posts_to_dispatch_endpoint(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 204
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        config = WorkflowDispatchConfig(
            repository="owner/repo",
            workflow_file="pipeline.yml",
            ref="main",
            mode="tests",
            force_refresh_prices=False,
            token="token-123",
        )
        result = dispatch_workflow(config)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["status_code"], 204)
        mock_post.assert_called_once()
        self.assertIn("/repos/owner/repo/actions/workflows/pipeline.yml/dispatches", mock_post.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
