"""Self-tests for the opaque-box E2E test harness and test doubles."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest

from e2e_tests.harness import (
    AppProcessHarness,
    FakeLlamaServer,
    build_mock_llama_exe,
    build_mock_python_exe,
    find_free_port,
    http_request,
    is_pid_alive,
    is_port_open,
    make_recording_plugin_source,
    read_jsonl,
    read_mock_llama_args_log,
    write_dummy_gguf,
    write_t2i_plugin,
)


class TestInfraSelf(unittest.TestCase):
    def test_01_find_free_port_is_bindable(self) -> None:
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
            self.assertEqual(sock.getsockname()[1], port)

    def test_02_fake_llama_server_health_and_chat_capture(self) -> None:
        with FakeLlamaServer(health_sequence=[503, 503, 200]) as server:
            server.set_reply_fn(
                lambda body, idx: f"turn-{idx}:{body['messages'][-1]['content']}"
            )
            h1 = http_request(f"{server.base_url}/health")
            h2 = http_request(f"{server.base_url}/health")
            h3 = http_request(f"{server.base_url}/health")
            self.assertEqual([h1.status, h2.status, h3.status], [503, 503, 200])
            self.assertEqual(h3.data["status"], "ok")
            self.assertEqual(len(server.health_requests), 3)

            payload = {
                "messages": [
                    {"role": "system", "content": "You are Firefly."},
                    {"role": "user", "content": "你好"},
                ],
                "temperature": 0.7,
            }
            resp = http_request(
                f"{server.base_url}/v1/chat/completions",
                method="POST",
                json_body=payload,
            )
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.data["choices"][0]["message"]["content"], "turn-1:你好")
            reqs = server.wait_for_requests(1, timeout=2.0)
            self.assertEqual(len(reqs), 1)
            self.assertEqual(reqs[0]["body"]["messages"][0]["content"], "You are Firefly.")

    def test_03_fake_llama_server_fault_modes_and_interruptible_delay(self) -> None:
        with FakeLlamaServer() as server:
            server.enqueue_response(mode="http_500", error_status=500)
            server.enqueue_response(mode="malformed_json")
            server.enqueue_response(mode="empty_choices")
            server.enqueue_response(mode="drop_connection")

            r_500 = http_request(
                f"{server.base_url}/v1/chat/completions",
                method="POST",
                json_body={"messages": []},
            )
            self.assertEqual(r_500.status, 500)

            r_bad = http_request(
                f"{server.base_url}/v1/chat/completions",
                method="POST",
                json_body={"messages": []},
            )
            self.assertEqual(r_bad.status, 200)
            self.assertIsNone(r_bad.data)
            self.assertEqual(r_bad.text, "{broken-json")

            r_empty = http_request(
                f"{server.base_url}/v1/chat/completions",
                method="POST",
                json_body={"messages": []},
            )
            self.assertEqual(r_empty.status, 200)
            self.assertEqual(r_empty.data["choices"], [])

            with self.assertRaises(Exception):
                http_request(
                    f"{server.base_url}/v1/chat/completions",
                    method="POST",
                    json_body={"messages": []},
                    timeout=2.0,
                )

            # Verify stop() interrupts a long response delay promptly
            server.set_mode("ok", delay_sec=10.0)
            import threading

            bg_errors: list[Exception] = []

            def _bg_call() -> None:
                try:
                    http_request(
                        f"{server.base_url}/v1/chat/completions",
                        method="POST",
                        json_body={"messages": []},
                        timeout=3.0,
                    )
                except Exception as exc:
                    bg_errors.append(exc)

            t = threading.Thread(target=_bg_call, daemon=True)
            t.start()
            server.wait_for_requests(5, timeout=2.0)
            t0 = time.monotonic()
            server.stop()
            self.assertLess(time.monotonic() - t0, 2.0)
            t.join(timeout=2.0)
            self.assertTrue(len(bg_errors) == 1)

    def test_04_build_mock_llama_exe_loading_chat_and_args_log(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_exe_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))

        exe_path = build_mock_llama_exe(tmp, "llama-server.exe")
        self.assertTrue(exe_path.is_file())
        args_log = tmp / "args.jsonl"
        gguf = write_dummy_gguf(tmp / "model.gguf")
        port = find_free_port()

        env = os.environ.copy()
        env["MOCK_LLAMA_ARGS_LOG"] = str(args_log)
        env["MOCK_LLAMA_MODE"] = "loading_then_ready:2"

        proc = subprocess.Popen(
            [
                str(exe_path),
                "-m",
                str(gguf),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-c",
                "4096",
                "-ngl",
                "12",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            deadline = time.monotonic() + 5.0
            statuses: list[int] = []
            while time.monotonic() < deadline:
                try:
                    resp = http_request(f"http://127.0.0.1:{port}/health", timeout=1.0)
                    statuses.append(resp.status)
                    if resp.status == 200:
                        break
                except Exception:
                    time.sleep(0.05)
            self.assertEqual(statuses, [503, 503, 200])

            chat_resp = http_request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                method="POST",
                json_body={"messages": [{"role": "user", "content": "hi"}]},
                timeout=3.0,
            )
            self.assertEqual(chat_resp.status, 200)
            self.assertIn("Mock C# llama-server reply", chat_resp.data["choices"][0]["message"]["content"])

            logged = read_mock_llama_args_log(args_log)
            self.assertEqual(len(logged), 1)
            self.assertIn("-m", logged[0])
            self.assertIn(str(gguf), logged[0])
            self.assertIn(str(port), logged[0])
        finally:
            proc.terminate()
            proc.wait(timeout=3.0)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    def test_05_build_mock_llama_exe_crash_mode(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_crash_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        exe_path = build_mock_llama_exe(tmp, "server.exe")
        env = os.environ.copy()
        env["MOCK_LLAMA_MODE"] = "crash"
        proc = subprocess.run(
            [str(exe_path), "--port", str(find_free_port())],
            capture_output=True,
            text=True,
            env=env,
            timeout=5.0,
        )
        self.assertEqual(proc.returncode, 2)

    def test_06_write_dummy_gguf_header_and_size(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_gguf_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        gguf = write_dummy_gguf(tmp / "nested" / "tiny.gguf")
        raw = gguf.read_bytes()
        self.assertEqual(len(raw), 64)
        self.assertEqual(raw[:8], b"GGUF\x03\x00\x00\x00")

    def test_07_dropin_recording_plugin_patterns_without_backend_import(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_plug_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        log_path = tmp / "t2i_log.jsonl"

        p1 = write_t2i_plugin(
            tmp,
            "rec_plugin.py",
            make_recording_plugin_source("rec_plugin", log_path, use_register=False),
        )
        p2 = write_t2i_plugin(
            tmp,
            "reg_plugin.py",
            make_recording_plugin_source("reg_plugin", log_path, use_register=True),
        )
        self.assertNotIn("backend", p1.read_text(encoding="utf-8"))
        self.assertNotIn("backend", p2.read_text(encoding="utf-8"))

        spec1 = importlib.util.spec_from_file_location("e2e_self_p1", p1)
        mod1 = importlib.util.module_from_spec(spec1)
        assert spec1 and spec1.loader
        spec1.loader.exec_module(mod1)

        ctx = types.SimpleNamespace(
            session_id="s-1",
            turn_index=2,
            timestamp_utc="2026-11-20T15:00:00Z",
            character_name="Firefly",
            skill_source="firefly-skill",
            user_message="看烟花吗？",
            assistant_reply="好呀。",
            recent_messages=({"role": "user", "content": "看烟花吗？"},),
            scene_prompt_hint="fireworks",
            metadata={"mode": "test"},
        )
        res1 = mod1.PLUGIN.on_turn_completed(ctx, {"style": "anime"})
        self.assertTrue(dataclasses.is_dataclass(res1))
        d1 = dataclasses.asdict(res1)
        self.assertEqual(d1["status"], "stub_ok")
        self.assertEqual(d1["plugin_name"], "rec_plugin")
        self.assertTrue(d1["invoked"])

        spec2 = importlib.util.spec_from_file_location("e2e_self_p2", p2)
        mod2 = importlib.util.module_from_spec(spec2)
        assert spec2 and spec2.loader
        spec2.loader.exec_module(mod2)
        registered: list[object] = []
        mod2.register(types.SimpleNamespace(register=registered.append))
        self.assertEqual(len(registered), 1)
        res2 = registered[0].on_turn_completed(ctx, {"seed": 7})
        self.assertEqual(dataclasses.asdict(res2)["plugin_name"], "reg_plugin")

        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["character_name"], "Firefly")
        self.assertEqual(entries[1]["plugin_settings"], {"seed": 7})

    def test_08_app_process_harness_fails_fast_when_target_missing(self) -> None:
        empty_repo = Path(tempfile.mkdtemp(prefix="moris_empty_repo_"))
        self.addCleanup(lambda: shutil.rmtree(empty_repo, ignore_errors=True))

        harness = AppProcessHarness(repo_root=empty_repo)
        self.addCleanup(harness.stop)
        with self.assertRaises(FileNotFoundError) as ctx:
            harness.start()
        self.assertIn("[E2E TARGET MISSING]", str(ctx.exception))
        self.assertFalse(harness.temp_dir.exists())

        # Also verify missing start.bat fails fast when app.py exists
        (empty_repo / "app.py").write_text("# stub\n", encoding="utf-8")
        bat_harness = AppProcessHarness(repo_root=empty_repo, use_bat_launcher=True)
        self.addCleanup(bat_harness.stop)
        with self.assertRaises(FileNotFoundError) as bat_ctx:
            bat_harness.start()
        self.assertIn("[E2E TARGET MISSING]", str(bat_ctx.exception))
        self.assertIn("start.bat", str(bat_ctx.exception))
        self.assertFalse(bat_harness.temp_dir.exists())

    def test_09_bat_launcher_teardown_kills_child_tree_no_deadlock(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_bat_stop_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        fake_repo = tmp / "repo"
        fake_repo.mkdir()
        pid_file = tmp / "child_pid.txt"

        (fake_repo / "app.py").write_text(
            f"""import argparse, http.server, json, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--config", default="")
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        return
    def do_GET(self):
        body = json.dumps({{"status": "ok"}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
""",
            encoding="utf-8",
        )
        (fake_repo / "start.bat").write_text(
            f'@echo off\r\n"{sys.executable}" "%~dp0app.py" %*\r\n',
            encoding="utf-8",
        )

        harness = AppProcessHarness(repo_root=fake_repo, use_bat_launcher=True)
        self.addCleanup(harness.stop)
        harness.start()

        child_pid = int(pid_file.read_text(encoding="utf-8").strip())
        self.assertTrue(is_pid_alive(child_pid))
        self.assertTrue(is_port_open("127.0.0.1", harness.port))

        t0 = time.monotonic()
        harness.stop()
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 5.0)
        self.assertFalse(is_pid_alive(child_pid))
        self.assertFalse(is_port_open("127.0.0.1", harness.port))

    def test_10_bat_launcher_handles_spaces_and_unicode_in_bat_and_config_paths(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="moris_self_bat_quotes_"))
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        space_repo = tmp / "moris roleplay space"
        space_repo.mkdir()
        uni_cfg_dir = tmp / "角色 配置"
        uni_cfg_dir.mkdir()
        cfg_path = uni_cfg_dir / "my config.json"

        (space_repo / "app.py").write_text(
            """import argparse, http.server, json

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--config", default="")
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        return
    def do_GET(self):
        body = json.dumps({"status": "ok", "received_config": args.config}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
""",
            encoding="utf-8",
        )
        (space_repo / "start.bat").write_text(
            f'@echo off\r\n"{sys.executable}" "%~dp0app.py" %*\r\n',
            encoding="utf-8",
        )

        # Quick check of build_mock_python_exe too
        mock_bin = tmp / "mock_bin"
        mock_py = build_mock_python_exe(mock_bin, "python.exe", "Python 3.9.0", 1)
        self.assertTrue(mock_py.is_file())

        harness = AppProcessHarness(
            repo_root=space_repo,
            use_bat_launcher=True,
            config_path=cfg_path,
            write_config=False,
        )
        self.addCleanup(harness.stop)
        harness.start()

        health = harness.get_json("/api/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(health.data["received_config"], str(cfg_path))
        harness.stop()


if __name__ == "__main__":
    unittest.main()
