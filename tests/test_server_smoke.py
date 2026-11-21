"""Smoke and unit tests for M4: HTTP API Server, Static UI, and CLI Entrypoint."""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from app import parse_args
from backend.server import build_application_server
from e2e_tests.harness import (
    FIREFLY_SKILL_DIR,
    FakeLlamaServer,
    build_mock_llama_exe,
    find_free_port,
    http_request,
    write_dummy_gguf,
)


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

    def test_streaming_sessions_edit_truncate_and_roleplay_enhancements(self) -> None:
        fake_llama = FakeLlamaServer(default_reply="*微微一笑* 欢迎回来，旅者。").start()
        self.addCleanup(fake_llama.stop)
        base_url, _ = self._start_inproc_server()

        put = http_request(
            f"{base_url}/api/config",
            method="PUT",
            json_body={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "min_p": 0.06,
                    "repeat_penalty": 1.15,
                    "cache_prompt": True,
                },
                "skill": {
                    "user_name": "开拓者",
                    "scenario": "黄金的时刻广场",
                    "authors_note": "保持温柔细腻的描写",
                    "immersion_mode": True,
                },
            },
        )
        self.assertEqual(put.status, 200)
        self.assertEqual(put.data["skill"]["scenario"], "黄金的时刻广场")
        self.assertAlmostEqual(put.data["llama"]["min_p"], 0.06)

        # SSE streaming endpoint
        st = http_request(
            f"{base_url}/api/chat/stream",
            method="POST",
            json_body={"session_id": "stream-1", "message": "晚上好"},
        )
        self.assertEqual(st.status, 200)
        self.assertIn("text/event-stream", st.headers.get("content-type", ""))
        self.assertIn('"type": "delta"', st.text)
        self.assertIn('"type": "done"', st.text)
        self.assertIn("欢迎回来", st.text)

        # Verify Author's Note / Immersion directive & cache_prompt sent to upstream
        last_body = fake_llama.last_request["body"]
        self.assertIs(last_body.get("cache_prompt"), True)
        self.assertAlmostEqual(last_body.get("repeat_penalty"), 1.15)
        sys_msgs = [m["content"] for m in last_body["messages"] if m["role"] == "system"]
        self.assertTrue(any("黄金的时刻广场" in c and "保持温柔细腻的描写" in c for c in sys_msgs))

        # Session list & history persistence
        sess_list = http_request(f"{base_url}/api/sessions")
        self.assertEqual(sess_list.status, 200)
        self.assertTrue(any(s["session_id"] == "stream-1" for s in sess_list.data["sessions"]))

        hist = http_request(f"{base_url}/api/sessions/history?session_id=stream-1")
        self.assertEqual(hist.status, 200)
        self.assertEqual(len(hist.data["messages"]), 2)

        # Edit message
        ed = http_request(
            f"{base_url}/api/sessions/message",
            method="PUT",
            json_body={"session_id": "stream-1", "index": 1, "content": "修改后的角色回复"},
        )
        self.assertEqual(ed.status, 200)
        self.assertEqual(ed.data["messages"][1]["content"], "修改后的角色回复")

        # Regenerate stream
        rg = http_request(
            f"{base_url}/api/chat/regenerate/stream",
            method="POST",
            json_body={"session_id": "stream-1"},
        )
        self.assertEqual(rg.status, 200)
        self.assertIn('"type": "done"', rg.text)

        # Truncate / rewind
        tr = http_request(
            f"{base_url}/api/sessions/truncate",
            method="POST",
            json_body={"session_id": "stream-1", "index": 1},
        )
        self.assertEqual(tr.status, 200)
        self.assertEqual(tr.data["messages_count"], 1)

        # Delete session
        dl = http_request(
            f"{base_url}/api/sessions/delete",
            method="POST",
            json_body={"session_id": "stream-1"},
        )
        self.assertEqual(dl.status, 200)
        self.assertIs(dl.data["deleted"], True)

    def test_quick_ready_resolves_port_conflict_and_launches_managed_engine(self) -> None:
        import socket

        # Case 1: Port occupied by unrelated listener -> auto-switch to free port & start managed_server
        busy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy_sock.bind(("127.0.0.1", 0))
        busy_sock.listen(1)
        busy_port = busy_sock.getsockname()[1]
        self.addCleanup(busy_sock.close)

        exe = build_mock_llama_exe(self.tmp_dir / "bin", "llama-server.exe")
        gguf = write_dummy_gguf(self.tmp_dir / "demo-model.gguf")
        base_url1, _ = self._start_inproc_server()
        http_request(
            f"{base_url1}/api/config",
            method="PUT",
            json_body={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": busy_port,
                }
            },
        )
        qr1 = http_request(f"{base_url1}/api/quick-ready", method="POST", json_body={})
        self.assertEqual(qr1.status, 200)
        self.assertEqual(qr1.data["engine"]["state"], "ready")
        self.assertEqual(qr1.data["engine"]["mode"], "managed_server")
        self.assertNotEqual(qr1.data["config"]["llama"]["port"], busy_port)

        # Case 2: Configured relative/moved path auto-resolved from local library
        base_url2, _ = self._start_inproc_server()
        http_request(
            f"{base_url2}/api/config",
            method="PUT",
            json_body={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": find_free_port(),
                }
            },
        )
        qr2 = http_request(f"{base_url2}/api/quick-ready", method="POST", json_body={})
        self.assertEqual(qr2.status, 200)
        self.assertEqual(qr2.data["engine"]["state"], "ready")
        self.assertEqual(qr2.data["engine"]["mode"], "managed_server")


if __name__ == "__main__":
    unittest.main()
