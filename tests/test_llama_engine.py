"""Unit tests for backend.llama_engine using local HTTP and subprocess test doubles."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

from backend.llama_engine import (
    EngineError,
    EngineStatus,
    LlamaCliEngine,
    LlamaServerEngine,
    create_engine,
    probe_tcp_port_available,
    resolve_executable,
    validate_gguf_model,
)

_DUMMY_SERVER_SCRIPT = """\
import http.server
import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log_path = os.environ.get("TEST_ARGV_LOG")
if log_path:
    Path(log_path).write_text(json.dumps(args), encoding="utf-8")

mode = os.environ.get("TEST_SERVER_MODE", "ok")
if mode == "crash":
    sys.stderr.write("fatal: mock llama-server crash\\n")
    sys.stderr.flush()
    sys.exit(2)

if mode == "hang":
    time.sleep(30)
    sys.exit(0)

host = "127.0.0.1"
port = 0
i = 0
while i < len(args):
    if args[i] == "--host" and i + 1 < len(args):
        host = args[i + 1]
        i += 2
    elif args[i] == "--port" and i + 1 < len(args):
        port = int(args[i + 1])
        i += 2
    else:
        i += 1

loading_remaining = 0
if mode.startswith("loading_then_ready:"):
    loading_remaining = int(mode.split(":", 1)[1])

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *a):
        return

    def do_GET(self):
        global loading_remaining
        if self.path == "/health":
            if mode == "always_loading":
                body = json.dumps({"status": "loading model"}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if loading_remaining > 0:
                loading_remaining -= 1
                body = json.dumps({"status": "loading model"}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req_data = json.loads(raw.decode("utf-8"))
            delay_raw = os.environ.get("TEST_CHAT_DELAY_SEC", "0")
            delay = float(delay_raw) if delay_raw else 0.0
            if delay > 0:
                time.sleep(delay)
            last_user = ""
            for msg in reversed(req_data.get("messages", [])):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            resp = {
                "id": "chatcmpl-test",
                "model": "mock-gguf",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"mock-reply:{last_user}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

class ReusableServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

srv = ReusableServer((host, port), Handler)
srv.serve_forever()
"""

_DUMMY_CLI_SCRIPT = """\
import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log_path = os.environ.get("TEST_CLI_ARGV_LOG")
if log_path:
    Path(log_path).write_text(json.dumps(args), encoding="utf-8")

mode = os.environ.get("TEST_CLI_MODE", "ok")
if mode == "fail":
    sys.stderr.write("cli failure detail\\n")
    sys.stderr.flush()
    sys.exit(3)

if mode == "sleep":
    time.sleep(5)
    sys.exit(0)

if mode == "empty":
    sys.exit(0)

prompt = ""
i = 0
while i < len(args):
    if args[i] == "-p" and i + 1 < len(args):
        prompt = args[i + 1]
        i += 2
    elif args[i] == "-f" and i + 1 < len(args):
        prompt = Path(args[i + 1]).read_text(encoding="utf-8")
        i += 2
    else:
        i += 1

if mode == "echo_prompt":
    sys.stdout.write(prompt + "CLI echoed reply<|im_end|>trailing")
else:
    sys.stdout.write(f"cli-reply:len={len(prompt)}")
sys.stdout.flush()
"""


def _alloc_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LocalFakeLlamaServer:
    """In-process ThreadingHTTPServer double for external_server tests."""

    def __init__(self) -> None:
        self.health_modes: list[str] = []
        self.default_health_mode: str = "ok"
        self.chat_mode: str = "ok"
        self.chat_delay_sec: float = 0.0
        self.health_delay_sec: float = 0.0
        self.recorded_requests: list[dict[str, Any]] = []
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                if parent.health_delay_sec > 0:
                    time.sleep(parent.health_delay_sec)
                mode = parent.health_modes.pop(0) if parent.health_modes else parent.default_health_mode
                if mode == "503":
                    body = json.dumps({"status": "loading model"}).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if mode == "200_loading":
                    body = json.dumps({"status": "loading"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps({"status": "ok"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw_req = self.rfile.read(length)
                req_json = json.loads(raw_req.decode("utf-8"))
                parent.recorded_requests.append(req_json)

                if parent.chat_delay_sec > 0:
                    time.sleep(parent.chat_delay_sec)

                mode = parent.chat_mode
                if mode == "500":
                    body = b"internal error"
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if mode == "503":
                    body = b'{"error":"loading"}'
                    self.send_response(503)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if mode == "malformed_json":
                    body = b"{not-json"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if mode == "empty_choices":
                    body = json.dumps({"choices": []}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if mode == "drop":
                    self.connection.close()
                    return

                resp = {
                    "model": "ext-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "external-reply"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                }
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

        class _Server(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

            def handle_error(self, request: Any, client_address: Any) -> None:
                return

        self._server = _Server(("127.0.0.1", 0), _Handler)
        self.port = int(self._server.server_address[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def _write_cmd_double(dir_path: Path, name: str, py_name: str, py_body: str) -> Path:
    py_path = dir_path / py_name
    py_path.write_text(py_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = dir_path / f"{name}.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" "{py_path}" %*\r\n',
            encoding="utf-8",
        )
    else:
        wrapper = dir_path / name
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{py_path}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return wrapper


class LlamaEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self._engines_to_cleanup: list[LlamaServerEngine] = []
        self._servers_to_cleanup: list[_LocalFakeLlamaServer] = []

    def tearDown(self) -> None:
        for eng in self._engines_to_cleanup:
            try:
                eng.shutdown()
            except Exception:
                pass
        for srv in self._servers_to_cleanup:
            try:
                srv.close()
            except Exception:
                pass
        self._tmp.cleanup()

    def _make_gguf(self, name: str = "model.gguf", content: bytes = b"GGUF_STUB") -> Path:
        p = self.tmp_dir / name
        p.write_bytes(content)
        return p

    # ------------------------------------------------------------------
    # Error, Status & Factory
    # ------------------------------------------------------------------

    def test_engine_error_shapes_and_status_copy(self) -> None:
        err = EngineError("ENGINE_PORT_CONFLICT", "busy", http_status=409, detail="port 8080")
        self.assertEqual(err.code, "ENGINE_PORT_CONFLICT")
        self.assertEqual(err.http_status, 409)
        self.assertEqual(err.to_dict(), {"code": "ENGINE_PORT_CONFLICT", "message": "busy", "detail": "port 8080"})
        self.assertEqual(err.to_error_payload(), {"error": err.to_dict()})

        err_pos = EngineError("MODEL_FILE_INVALID", "bad model", "missing.gguf")
        self.assertEqual(err_pos.http_status, 400)
        self.assertEqual(err_pos.detail, "missing.gguf")

        engine = LlamaServerEngine({"engine_mode": "external_server", "server_url": "http://127.0.0.1:9999"})
        s1 = engine.status()
        self.assertIsInstance(s1, EngineStatus)
        s1.state = "ready"
        self.assertEqual(engine.status().state, "stopped")
        self.assertEqual(engine.status().to_dict()["mode"], "external_server")

    def test_create_engine_modes_and_full_config_unwrap(self) -> None:
        self.assertIsInstance(create_engine({"engine_mode": "managed_server"}), LlamaServerEngine)
        self.assertIsInstance(create_engine({"engine_mode": "external_server"}), LlamaServerEngine)
        self.assertIsInstance(create_engine({"engine_mode": "cli"}), LlamaCliEngine)
        self.assertIsInstance(
            create_engine({"llama": {"engine_mode": "external_server", "server_url": "http://127.0.0.1:8080"}}),
            LlamaServerEngine,
        )
        with self.assertRaises(EngineError) as ctx:
            create_engine({"engine_mode": "unknown_mode"})
        self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")
        self.assertEqual(ctx.exception.http_status, 400)

    # ------------------------------------------------------------------
    # Binary & GGUF Validation
    # ------------------------------------------------------------------

    def test_binary_and_gguf_validation_errors(self) -> None:
        with self.assertRaises(EngineError) as ctx:
            resolve_executable("", ("llama-server.exe",))
        self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")

        with self.assertRaises(EngineError) as ctx:
            resolve_executable(str(self.tmp_dir / "missing.exe"), ("llama-server.exe",))
        self.assertEqual(ctx.exception.code, "ENGINE_BINARY_NOT_FOUND")

        empty_dir = self.tmp_dir / "empty_bin_dir"
        empty_dir.mkdir()
        with self.assertRaises(EngineError) as ctx:
            resolve_executable(str(empty_dir), ("llama-server.exe",))
        self.assertEqual(ctx.exception.code, "ENGINE_BINARY_NOT_FOUND")

        if os.name == "nt":
            txt_file = self.tmp_dir / "not_exe.txt"
            txt_file.write_text("hello", encoding="utf-8")
            with self.assertRaises(EngineError) as ctx:
                resolve_executable(str(txt_file), ("llama-server.exe",))
            self.assertEqual(ctx.exception.code, "ENGINE_BINARY_NOT_FOUND")

            # Directory discovery precedence: llama-server.exe > server.exe > llama-server.cmd
            bin_dir = self.tmp_dir / "bin_prec"
            bin_dir.mkdir()
            cmd_path = _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
            found_cmd = resolve_executable(
                str(bin_dir), ("llama-server.exe", "server.exe", "llama-server.cmd")
            )
            self.assertEqual(found_cmd, cmd_path.resolve())

            server_exe = bin_dir / "server.exe"
            server_exe.write_bytes(b"MZ")
            found_srv = resolve_executable(
                str(bin_dir), ("llama-server.exe", "server.exe", "llama-server.cmd")
            )
            self.assertEqual(found_srv, server_exe.resolve())

            llama_exe = bin_dir / "llama-server.exe"
            llama_exe.write_bytes(b"MZ")
            found_llama = resolve_executable(
                str(bin_dir), ("llama-server.exe", "server.exe", "llama-server.cmd")
            )
            self.assertEqual(found_llama, llama_exe.resolve())

        # GGUF validation
        for bad_name, setup_fn in (
            ("missing", lambda: self.tmp_dir / "does_not_exist.gguf"),
            ("directory", lambda: empty_dir),
            ("wrong_ext", lambda: self._make_gguf("model.bin", b"123")),
            ("zero_bytes", lambda: self._make_gguf("empty.gguf", b"")),
        ):
            with self.subTest(bad_name=bad_name):
                p = setup_fn()
                with self.assertRaises(EngineError) as ctx:
                    validate_gguf_model(str(p))
                self.assertEqual(ctx.exception.code, "MODEL_FILE_INVALID")
                self.assertEqual(ctx.exception.http_status, 400)

        valid_gguf = self._make_gguf("good.GGUF", b"data")
        self.assertEqual(validate_gguf_model(str(valid_gguf)), valid_gguf.resolve())

    # ------------------------------------------------------------------
    # External Server Mode
    # ------------------------------------------------------------------

    def test_external_server_health_and_chat_and_faults(self) -> None:
        # Empty / invalid URL -> CONFIG_INCOMPLETE
        bad_eng = LlamaServerEngine({"engine_mode": "external_server", "server_url": ""})
        with self.assertRaises(EngineError) as ctx:
            bad_eng.ensure_ready()
        self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")
        self.assertEqual(ctx.exception.http_status, 400)

        srv = _LocalFakeLlamaServer()
        self._servers_to_cleanup.append(srv)

        eng = LlamaServerEngine(
            {
                "engine_mode": "external_server",
                "server_url": srv.url,
                "startup_timeout_sec": 0.4,
                "request_timeout_sec": 0.4,
                "temperature": 0.7,
                "top_p": 0.85,
                "max_tokens": 256,
            }
        )

        # 503 loading -> ENGINE_LOADING (503)
        srv.health_modes = ["503", "200_loading"]
        with self.assertRaises(EngineError) as ctx:
            eng.ensure_ready()
        self.assertEqual(ctx.exception.code, "ENGINE_LOADING")
        self.assertEqual(ctx.exception.http_status, 503)

        with self.assertRaises(EngineError) as ctx:
            eng.ensure_ready()
        self.assertEqual(ctx.exception.code, "ENGINE_LOADING")
        self.assertEqual(ctx.exception.http_status, 503)

        # Healthy -> ready + chat_completion
        eng.ensure_ready()
        self.assertEqual(eng.status().state, "ready")

        res = eng.chat_completion(
            [{"role": "system", "content": "You are Firefly."}, {"role": "user", "content": "Hi"}],
            {"temperature": 0.55},
        )
        self.assertEqual(res["reply"], "external-reply")
        self.assertEqual(res["usage"]["total_tokens"], 14)
        self.assertEqual(srv.recorded_requests[-1]["temperature"], 0.55)
        self.assertEqual(srv.recorded_requests[-1]["top_p"], 0.85)
        self.assertEqual(srv.recorded_requests[-1]["max_tokens"], 256)

        # Fault matrix on /v1/chat/completions
        for fault_mode, expected_code, expected_status in (
            ("500", "ENGINE_UPSTREAM_ERROR", 502),
            ("503", "ENGINE_LOADING", 503),
            ("malformed_json", "ENGINE_UPSTREAM_ERROR", 502),
            ("empty_choices", "ENGINE_UPSTREAM_ERROR", 502),
            ("drop", "ENGINE_UPSTREAM_ERROR", 502),
        ):
            with self.subTest(fault_mode=fault_mode):
                srv.chat_mode = fault_mode
                with self.assertRaises(EngineError) as ctx:
                    eng.chat_completion([{"role": "user", "content": "test"}])
                self.assertEqual(ctx.exception.code, expected_code)
                self.assertEqual(ctx.exception.http_status, expected_status)
                self.assertEqual(eng.status().state, "error")

        # Timeout fault on slow response
        srv.chat_mode = "ok"
        srv.chat_delay_sec = 0.8
        with self.assertRaises(EngineError) as ctx:
            eng.chat_completion([{"role": "user", "content": "slow"}])
        self.assertEqual(ctx.exception.code, "ENGINE_TIMEOUT")
        self.assertEqual(ctx.exception.http_status, 504)

        # Recovery on next healthy request
        srv.chat_delay_sec = 0.0
        recovered = eng.chat_completion([{"role": "user", "content": "back"}])
        self.assertEqual(recovered["reply"], "external-reply")
        self.assertEqual(eng.status().state, "ready")

        # Dead port -> ENGINE_UPSTREAM_ERROR (502)
        dead_port = _alloc_free_port()
        dead_eng = LlamaServerEngine(
            {"engine_mode": "external_server", "server_url": f"http://127.0.0.1:{dead_port}"}
        )
        with self.assertRaises(EngineError) as ctx:
            dead_eng.chat_completion([{"role": "user", "content": "hello"}])
        self.assertEqual(ctx.exception.code, "ENGINE_UPSTREAM_ERROR")
        self.assertEqual(ctx.exception.http_status, 502)

    # ------------------------------------------------------------------
    # Managed Server Mode
    # ------------------------------------------------------------------

    def test_managed_server_incomplete_and_port_conflict(self) -> None:
        eng_empty = LlamaServerEngine({"engine_mode": "managed_server"})
        with self.assertRaises(EngineError) as ctx:
            eng_empty.ensure_ready()
        self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")

        bin_dir = self.tmp_dir / "bin"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            busy_port = int(holder.getsockname()[1])

            eng_conflict = LlamaServerEngine(
                {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(bin_dir),
                    "model_path": str(model),
                    "host": "127.0.0.1",
                    "port": busy_port,
                }
            )
            with self.assertRaises(EngineError) as ctx:
                eng_conflict.ensure_ready()
            self.assertEqual(ctx.exception.code, "ENGINE_PORT_CONFLICT")
            self.assertEqual(ctx.exception.http_status, 409)

    def test_managed_server_lifecycle_argv_chat_and_clean_shutdown(self) -> None:
        bin_dir = self.tmp_dir / "bin_ok"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()
        argv_log = self.tmp_dir / "argv.json"
        port = _alloc_free_port()

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "host": "127.0.0.1",
                "port": port,
                "ctx_size": 4096,
                "n_gpu_layers": 16,
                "startup_timeout_sec": 10.0,
                "request_timeout_sec": 10.0,
            }
        )
        self._engines_to_cleanup.append(eng)

        with patch.dict(
            os.environ,
            {"TEST_SERVER_MODE": "loading_then_ready:2", "TEST_ARGV_LOG": str(argv_log)},
        ):
            # Calling chat_completion from stopped state auto-starts managed server
            reply = eng.chat_completion([{"role": "user", "content": "Hello Firefly"}])

        self.assertEqual(reply["reply"], "mock-reply:Hello Firefly")
        st = eng.status()
        self.assertEqual(st.state, "ready")
        self.assertIsNotNone(st.pid)
        self.assertGreater(st.pid, 0)

        recorded_argv = json.loads(argv_log.read_text(encoding="utf-8"))
        self.assertEqual(
            recorded_argv,
            [
                "-m",
                str(model.resolve()),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-c",
                "4096",
                "-ngl",
                "16",
            ],
        )

        # Shutdown kills process tree and frees TCP port immediately
        eng.shutdown()
        self.assertEqual(eng.status().state, "stopped")
        self.assertIsNone(eng.status().pid)
        probe_tcp_port_available("127.0.0.1", port)
        # Idempotent second shutdown
        eng.shutdown()

    def test_managed_server_mid_session_exit_raises_process_exited(self) -> None:
        bin_dir = self.tmp_dir / "bin_mid_exit"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()
        port = _alloc_free_port()

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "port": port,
                "startup_timeout_sec": 8.0,
            }
        )
        self._engines_to_cleanup.append(eng)
        eng.ensure_ready()
        pid = eng.status().pid
        self.assertIsNotNone(pid)

        # Simulate unexpected external termination of the managed process between turns
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        else:
            eng._proc.kill()  # type: ignore[union-attr]
            eng._proc.wait(timeout=2.0)  # type: ignore[union-attr]

        with self.assertRaises(EngineError) as ctx:
            eng.chat_completion([{"role": "user", "content": "after crash"}])
        self.assertEqual(ctx.exception.code, "ENGINE_PROCESS_EXITED")
        self.assertEqual(ctx.exception.http_status, 502)
        self.assertEqual(eng.status().state, "error")

    def test_managed_server_crash_loading_timeout_and_nonblocking_status(self) -> None:
        bin_dir = self.tmp_dir / "bin_faults"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()

        # 1. Premature exit -> ENGINE_PROCESS_EXITED (502) with stderr tail
        port_crash = _alloc_free_port()
        eng_crash = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "port": port_crash,
                "startup_timeout_sec": 5.0,
            }
        )
        self._engines_to_cleanup.append(eng_crash)
        with patch.dict(os.environ, {"TEST_SERVER_MODE": "crash"}):
            with self.assertRaises(EngineError) as ctx:
                eng_crash.ensure_ready()
        self.assertEqual(ctx.exception.code, "ENGINE_PROCESS_EXITED")
        self.assertEqual(ctx.exception.http_status, 502)
        self.assertIn("mock llama-server crash", ctx.exception.detail)
        self.assertEqual(eng_crash.status().state, "error")

        # 2. Always 503 beyond startup_timeout_sec -> ENGINE_LOADING (503) + cleanup
        port_load = _alloc_free_port()
        eng_load = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "port": port_load,
                "startup_timeout_sec": 0.6,
            }
        )
        self._engines_to_cleanup.append(eng_load)
        with patch.dict(os.environ, {"TEST_SERVER_MODE": "always_loading"}):
            with self.assertRaises(EngineError) as ctx:
                eng_load.ensure_ready()
        self.assertEqual(ctx.exception.code, "ENGINE_LOADING")
        self.assertEqual(ctx.exception.http_status, 503)
        probe_tcp_port_available("127.0.0.1", port_load)

        # 3. Hang (never binds) -> ENGINE_TIMEOUT (504) + non-blocking status() check
        port_hang = _alloc_free_port()
        eng_hang = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "port": port_hang,
                "startup_timeout_sec": 0.7,
            }
        )
        self._engines_to_cleanup.append(eng_hang)
        observed_states: list[str] = []

        def _poll_status() -> None:
            for _ in range(10):
                observed_states.append(eng_hang.status().state)
                time.sleep(0.05)

        with patch.dict(os.environ, {"TEST_SERVER_MODE": "hang"}):
            t = threading.Thread(target=_poll_status, daemon=True)
            t.start()
            with self.assertRaises(EngineError) as ctx:
                eng_hang.ensure_ready()
            t.join(timeout=2.0)

        self.assertEqual(ctx.exception.code, "ENGINE_TIMEOUT")
        self.assertEqual(ctx.exception.http_status, 504)
        self.assertIn("starting", observed_states)
        probe_tcp_port_available("127.0.0.1", port_hang)

    def test_managed_server_exe_precedence_with_mocked_popen(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows .exe precedence test")

        bin_dir = self.tmp_dir / "exe_dir"
        bin_dir.mkdir()
        (bin_dir / "llama-server.cmd").write_text("@echo off\r\n", encoding="utf-8")
        (bin_dir / "server.exe").write_bytes(b"MZ")
        llama_exe = bin_dir / "llama-server.exe"
        llama_exe.write_bytes(b"MZ")
        model = self._make_gguf()
        port = _alloc_free_port()

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "port": port,
                "startup_timeout_sec": 2.0,
            }
        )

        class _FakeProc:
            pid = 42424

            def __init__(self) -> None:
                import io
                self.stderr = io.StringIO("")
                self._terminated = False

            def poll(self) -> int | None:
                return 0 if self._terminated else None

            def terminate(self) -> None:
                self._terminated = True

            def wait(self, timeout: float | None = None) -> int:
                self._terminated = True
                return 0

        captured_cmd: list[str] = []

        def _fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
            captured_cmd.extend(cmd)
            return _FakeProc()

        with (
            patch("backend.llama_engine.subprocess.Popen", side_effect=_fake_popen),
            patch.object(LlamaServerEngine, "_probe_health", return_value=("ok", "")),
        ):
            eng.ensure_ready()
            self.assertEqual(Path(captured_cmd[0]), llama_exe.resolve())
            self.assertEqual(eng.status().pid, 42424)
            eng.shutdown()
            self.assertEqual(eng.status().state, "stopped")

    # ------------------------------------------------------------------
    # CLI Mode
    # ------------------------------------------------------------------

    def test_cli_engine_prompt_long_file_and_errors(self) -> None:
        bin_dir = self.tmp_dir / "cli_bin"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-cli", "cli.py", _DUMMY_CLI_SCRIPT)
        model = self._make_gguf()
        argv_log = self.tmp_dir / "cli_argv.json"

        eng = LlamaCliEngine(
            {
                "engine_mode": "cli",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "request_timeout_sec": 5.0,
            }
        )

        # Short prompt: mocked .exe uses -p; .cmd multi-line prompt safely uses -f
        with patch.dict(os.environ, {"TEST_CLI_MODE": "echo_prompt", "TEST_CLI_ARGV_LOG": str(argv_log)}):
            res = eng.chat_completion(
                [
                    {"role": "system", "content": "System rules"},
                    {"role": "user", "content": "Hello"},
                ]
            )
        self.assertEqual(res["reply"], "CLI echoed reply")
        if os.name == "nt":
            # Verify .exe binary uses inline -p for short prompts
            fake_exe = self.tmp_dir / "llama-cli.exe"
            fake_exe.write_bytes(b"MZ")
            eng_exe = LlamaCliEngine(
                {
                    "engine_mode": "cli",
                    "llama_cpp_path": str(fake_exe),
                    "model_path": str(model),
                }
            )
            class _FakeCliProc:
                pid = 99999
                returncode = 0
                stdout = None
                stderr = None

                def communicate(self, timeout: float | None = None) -> tuple[str, str]:
                    return ("inline reply", "")

                def kill(self) -> None:
                    return

            with patch("backend.llama_engine.subprocess.Popen", return_value=_FakeCliProc()) as mock_popen:
                res_exe = eng_exe.chat_completion([{"role": "user", "content": "Hi"}])
            self.assertEqual(res_exe["reply"], "inline reply")
            self.assertIn("-p", mock_popen.call_args[0][0])
            self.assertNotIn("-f", mock_popen.call_args[0][0])
        else:
            short_argv = json.loads(argv_log.read_text(encoding="utf-8"))
            self.assertIn("-p", short_argv)
            self.assertNotIn("-f", short_argv)

        # Long prompt (>8000 chars) uses -f temp file
        long_text = "x" * 8500
        with patch.dict(os.environ, {"TEST_CLI_MODE": "ok", "TEST_CLI_ARGV_LOG": str(argv_log)}):
            res_long = eng.chat_completion([{"role": "user", "content": long_text}])
        long_argv = json.loads(argv_log.read_text(encoding="utf-8"))
        self.assertIn("-f", long_argv)
        self.assertNotIn("-p", long_argv)
        tmp_prompt_path = Path(long_argv[long_argv.index("-f") + 1])
        self.assertFalse(tmp_prompt_path.exists())
        self.assertTrue(res_long["reply"].startswith("cli-reply:len="))

        # Non-zero exit -> ENGINE_PROCESS_EXITED (502)
        with patch.dict(os.environ, {"TEST_CLI_MODE": "fail"}):
            with self.assertRaises(EngineError) as ctx:
                eng.chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.code, "ENGINE_PROCESS_EXITED")
        self.assertEqual(ctx.exception.http_status, 502)
        self.assertIn("cli failure detail", ctx.exception.detail)

        # Empty output -> ENGINE_UPSTREAM_ERROR (502)
        with patch.dict(os.environ, {"TEST_CLI_MODE": "empty"}):
            with self.assertRaises(EngineError) as ctx:
                eng.chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.code, "ENGINE_UPSTREAM_ERROR")
        self.assertEqual(ctx.exception.http_status, 502)

        # Timeout -> ENGINE_TIMEOUT (504)
        eng_short_timeout = LlamaCliEngine(
            {
                "engine_mode": "cli",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "request_timeout_sec": 0.3,
            }
        )
        with patch.dict(os.environ, {"TEST_CLI_MODE": "sleep"}):
            with self.assertRaises(EngineError) as ctx:
                eng_short_timeout.chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.code, "ENGINE_TIMEOUT")
        self.assertEqual(ctx.exception.http_status, 504)

        eng.shutdown()
        self.assertEqual(eng.status().state, "stopped")

    # ------------------------------------------------------------------
    # Iteration 2 Regression Tests
    # ------------------------------------------------------------------

    def test_engine_error_default_http_status_with_kwarg_detail_and_ipv6_endpoint(self) -> None:
        err = EngineError("CONFIG_INCOMPLETE", "msg", detail="missing key")
        self.assertEqual(err.http_status, 400)
        self.assertEqual(err.detail, "missing key")

        err_unknown = EngineError("CUSTOM_UNKNOWN", "msg", detail="x")
        self.assertEqual(err_unknown.http_status, 502)

        eng_v6 = create_engine({"engine_mode": "managed_server", "host": "::1", "port": 8080})
        self.assertEqual(eng_v6.status().endpoint, "http://[::1]:8080")
        parsed = urllib.parse.urlparse(eng_v6.status().endpoint)
        self.assertEqual(parsed.hostname, "::1")
        self.assertEqual(parsed.port, 8080)

        eng_v6_any = create_engine({"engine_mode": "managed_server", "host": "::", "port": 8081})
        self.assertEqual(eng_v6_any.status().endpoint, "http://[::1]:8081")

        # external_server clamps non-positive startup_timeout_sec cleanly
        dead_port = _alloc_free_port()
        ext = LlamaServerEngine(
            {
                "engine_mode": "external_server",
                "server_url": f"http://127.0.0.1:{dead_port}",
                "startup_timeout_sec": 0,
            }
        )
        with self.assertRaises(EngineError) as ctx:
            ext.ensure_ready()
        self.assertIn(ctx.exception.code, ("ENGINE_TIMEOUT", "ENGINE_UPSTREAM_ERROR"))

    def test_windows_wildcard_0000_port_conflict_detected(self) -> None:
        bin_dir = self.tmp_dir / "bin_wildcard"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("0.0.0.0", 0))
            holder.listen(1)
            busy_port = int(holder.getsockname()[1])

            for check_host in ("0.0.0.0", "127.0.0.1"):
                with self.subTest(check_host=check_host):
                    with self.assertRaises(EngineError) as ctx:
                        probe_tcp_port_available(check_host, busy_port)
                    self.assertEqual(ctx.exception.code, "ENGINE_PORT_CONFLICT")
                    self.assertEqual(ctx.exception.http_status, 409)

            eng = LlamaServerEngine(
                {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(bin_dir),
                    "model_path": str(model),
                    "host": "127.0.0.1",
                    "port": busy_port,
                    "startup_timeout_sec": 1.0,
                }
            )
            with self.assertRaises(EngineError) as ctx:
                eng.ensure_ready()
            self.assertEqual(ctx.exception.code, "ENGINE_PORT_CONFLICT")
            self.assertEqual(ctx.exception.http_status, 409)

    def test_concurrent_managed_chat_completion_and_shutdown_race(self) -> None:
        bin_dir = self.tmp_dir / "bin_race"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()
        port = _alloc_free_port()

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "host": "127.0.0.1",
                "port": port,
                "startup_timeout_sec": 8.0,
                "request_timeout_sec": 5.0,
            }
        )
        self._engines_to_cleanup.append(eng)

        with patch.dict(os.environ, {"TEST_CHAT_DELAY_SEC": "0.35"}):
            eng.ensure_ready()
            self.assertEqual(eng.status().state, "ready")

            chat_outcome: dict[str, Any] = {}

            def _run_chat() -> None:
                try:
                    chat_outcome["res"] = eng.chat_completion(
                        [{"role": "user", "content": "slow-race"}]
                    )
                except Exception as exc:
                    chat_outcome["exc"] = exc

            t = threading.Thread(target=_run_chat, daemon=True)
            t.start()
            time.sleep(0.08)
            eng.shutdown()
            t.join(timeout=5.0)

        self.assertFalse(t.is_alive())
        if "exc" in chat_outcome:
            self.assertIsInstance(chat_outcome["exc"], EngineError)
            self.assertEqual(chat_outcome["exc"].code, "ENGINE_PROCESS_EXITED")
            self.assertEqual(chat_outcome["exc"].http_status, 502)
        final_status = eng.status()
        self.assertEqual(final_status.state, "stopped")
        self.assertIsNone(final_status.pid)

        # Also verify if HTTP completion returns right after shutdown(), status stays stopped
        port2 = _alloc_free_port()
        eng2 = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "host": "127.0.0.1",
                "port": port2,
                "startup_timeout_sec": 8.0,
                "request_timeout_sec": 5.0,
            }
        )
        self._engines_to_cleanup.append(eng2)
        eng2.ensure_ready()

        class _FakeResponse:
            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self) -> bytes:
                eng2.shutdown()
                return json.dumps(
                    {
                        "choices": [{"message": {"content": "late-ok"}, "finish_reason": "stop"}],
                        "usage": {},
                    }
                ).encode("utf-8")

        with patch("backend.llama_engine._open_url", return_value=_FakeResponse()):
            res2 = eng2.chat_completion([{"role": "user", "content": "late"}])
        self.assertEqual(res2["reply"], "late-ok")
        self.assertEqual(eng2.status().state, "stopped")
        self.assertIsNone(eng2.status().pid)

    def test_cli_cmd_timeout_tree_kill_is_fast_and_cleans_temp_files(self) -> None:
        bin_dir = self.tmp_dir / "cli_timeout_fast"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-cli", "cli.py", _DUMMY_CLI_SCRIPT)
        model = self._make_gguf()

        eng = LlamaCliEngine(
            {
                "engine_mode": "cli",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "request_timeout_sec": 0.4,
            }
        )

        tmp_root = Path(tempfile.gettempdir())
        before_temps = {p.name for p in tmp_root.glob("moris_cli_prompt_*.txt")}

        started = time.perf_counter()
        with patch.dict(os.environ, {"TEST_CLI_MODE": "sleep"}):
            with self.assertRaises(EngineError) as ctx:
                eng.chat_completion(
                    [
                        {"role": "system", "content": "Line1\nLine2"},
                        {"role": "user", "content": "hi"},
                    ]
                )
        elapsed = time.perf_counter() - started

        after_temps = {p.name for p in tmp_root.glob("moris_cli_prompt_*.txt")}
        self.assertEqual(ctx.exception.code, "ENGINE_TIMEOUT")
        self.assertEqual(ctx.exception.http_status, 504)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(after_temps - before_temps, set())

    def test_invalid_params_raise_config_incomplete_400_for_server_and_cli(self) -> None:
        srv = _LocalFakeLlamaServer()
        self._servers_to_cleanup.append(srv)
        srv_eng = LlamaServerEngine(
            {
                "engine_mode": "external_server",
                "server_url": srv.url,
                "startup_timeout_sec": 1.0,
                "request_timeout_sec": 1.0,
            }
        )

        bin_dir = self.tmp_dir / "cli_bad_params"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-cli", "cli.py", _DUMMY_CLI_SCRIPT)
        model = self._make_gguf()
        cli_eng = LlamaCliEngine(
            {
                "engine_mode": "cli",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
            }
        )

        bad_params_cases: list[Any] = [
            "bad",
            {"temperature": "bad"},
            {"temperature": None},
            {"temperature": True},
            {"temperature": float("nan")},
            {"top_p": float("inf")},
            {"max_tokens": 0},
            {"max_tokens": True},
        ]
        for eng in (srv_eng, cli_eng):
            for bad_p in bad_params_cases:
                with self.subTest(mode=eng.mode, bad_p=bad_p):
                    with self.assertRaises(EngineError) as ctx:
                        eng.chat_completion([{"role": "user", "content": "hi"}], params=bad_p)  # type: ignore[arg-type]
                    self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")
                    self.assertEqual(ctx.exception.http_status, 400)

        self.assertEqual(len(srv.recorded_requests), 0)

    def test_managed_server_weakref_finalize_cleans_up_on_gc(self) -> None:
        import gc

        bin_dir = self.tmp_dir / "bin_gc"
        bin_dir.mkdir()
        _write_cmd_double(bin_dir, "llama-server", "srv.py", _DUMMY_SERVER_SCRIPT)
        model = self._make_gguf()
        port = _alloc_free_port()

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(bin_dir),
                "model_path": str(model),
                "host": "127.0.0.1",
                "port": port,
                "startup_timeout_sec": 8.0,
            }
        )
        eng.ensure_ready()
        self.assertEqual(eng.status().state, "ready")
        del eng
        gc.collect()
        probe_tcp_port_available("127.0.0.1", port)


if __name__ == "__main__":
    unittest.main()
