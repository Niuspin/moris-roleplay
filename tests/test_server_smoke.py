"""Smoke and unit tests for M4: HTTP API Server, Static UI, and CLI Entrypoint."""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from app import parse_args
from backend.server import build_application_server
from e2e_tests.harness import FIREFLY_SKILL_DIR, FakeLlamaServer, find_free_port, http_request


class ServerSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="moris_test_srv_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_dir, ignore_errors=True))

    def _start_inproc_server(self) -> tuple[str, object]:
        cfg_path = self.tmp_dir / "config.json"
        srv = build_application_server(
            config_path=cfg_path,
            host="127.0.0.1",
            port=find_free_port(),
        )
        port = int(srv.server_address[1])
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()

        def _cleanup() -> None:
            try:
                srv.shutdown()
            except Exception:
                pass
            try:
                srv.chat_service.shutdown()
            except Exception:
                pass
            try:
                srv.server_close()
            except Exception:
                pass
            t.join(timeout=2.0)

        self.addCleanup(_cleanup)
        return f"http://127.0.0.1:{port}", srv

    def test_parse_args_defaults_and_flags(self) -> None:
        args = parse_args(["--host", "127.0.0.1", "--port", "7899", "--no-browser"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7899)
        self.assertTrue(args.no_browser)

    def test_static_and_rest_api_lifecycle(self) -> None:
        fake_llama = FakeLlamaServer(default_reply="流萤收到消息！").start()
        self.addCleanup(fake_llama.stop)
        base_url, _ = self._start_inproc_server()

        # Static assets
        self.assertEqual(http_request(f"{base_url}/").status, 200)
        self.assertEqual(http_request(f"{base_url}/static/style.css").status, 200)
        self.assertEqual(http_request(f"{base_url}/static/app.js").status, 200)
        self.assertIn(http_request(f"{base_url}/static/../app.py").status, (403, 404))

        # Health & config
        h = http_request(f"{base_url}/api/health")
        self.assertEqual(h.status, 200)
        self.assertEqual(h.data["status"], "ok")

        put = http_request(
            f"{base_url}/api/config",
            method="PUT",
            json_body={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "example_stub"},
            },
        )
        self.assertEqual(put.status, 200)

        # Skill load
        sk = http_request(
            f"{base_url}/api/skill/load",
            method="POST",
            json_body={"path": str(FIREFLY_SKILL_DIR)},
        )
        self.assertEqual(sk.status, 200)
        self.assertEqual(sk.data["character_name"], "firefly-hsr")

        # Chat turn + T2I hook
        ch = http_request(
            f"{base_url}/api/chat",
            method="POST",
            json_body={"session_id": "smoke-1", "message": "你好流萤"},
        )
        self.assertEqual(ch.status, 200)
        self.assertEqual(ch.data["reply"], "流萤收到消息！")
        self.assertEqual(ch.data["t2i_hook"]["status"], "stub_ok")
        self.assertIs(ch.data["t2i_hook"]["invoked"], True)


if __name__ == "__main__":
    unittest.main()
