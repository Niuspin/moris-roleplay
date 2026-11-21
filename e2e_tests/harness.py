"""Stdlib-only opaque-box E2E test harness and test doubles for Moris Roleplay."""

from __future__ import annotations

import atexit
import base64
from collections import deque
import csv
from dataclasses import dataclass
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent

SKILL_ASSETS_ROOT = Path(r"C:\Users\nzp18\Desktop\角色skill")
FIREFLY_SKILL_DIR = SKILL_ASSETS_ROOT / "流萤" / "firefly-skill"
AEMEATH_SKILL_DIR = SKILL_ASSETS_ROOT / "爱弥斯" / "Aemeath-skill"
YAE_MIKO_SKILL_FILE = SKILL_ASSETS_ROOT / "八重神子" / "SKILL.md"
SANDRONE_SKILL_FILE = SKILL_ASSETS_ROOT / "桑多涅" / "SKILL.md"
RAIDEN_SKILL_FILE = SKILL_ASSETS_ROOT / "雷电将军" / "SKILL.md"

CSC_EXE = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
EDGE_EXE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_MOCK_EXE_CACHE_DIR: Path | None = None
_MOCK_EXE_CACHE_PATH: Path | None = None
_MOCK_PYTHON_EXE_CACHE: dict[tuple[str, int], Path] = {}
_MOCK_EXE_CACHE_LOCK = threading.Lock()


def _cleanup_mock_exe_cache() -> None:
    global _MOCK_EXE_CACHE_DIR
    if _MOCK_EXE_CACHE_DIR is not None and _MOCK_EXE_CACHE_DIR.exists():
        shutil.rmtree(_MOCK_EXE_CACHE_DIR, ignore_errors=True)
        _MOCK_EXE_CACHE_DIR = None
    _MOCK_PYTHON_EXE_CACHE.clear()


atexit.register(_cleanup_mock_exe_cache)


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral TCP port and return the port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def require_target_file(path: Path) -> Path:
    """Fail fast with [E2E TARGET MISSING] when a target implementation file is absent."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"[E2E TARGET MISSING] {target} not found. Implementation Track M1-M4 must create it "
            "before running Tiers 1-4; run 'python -m unittest e2e_tests.test_infra_self -v' "
            "to verify E2E harness."
        )
    return target


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def is_pid_alive(pid: int) -> bool:
    """Check whether a Windows or POSIX PID is currently running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        for row in csv.reader(proc.stdout.splitlines()):
            if len(row) >= 2 and row[1].strip() == str(pid):
                return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-merged copy of ``base`` with ``override`` applied."""
    result: dict[str, Any] = {}
    for k, v in base.items():
        if isinstance(v, dict):
            result[k] = deep_merge(v, {})
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def write_dummy_gguf(path: Path) -> Path:
    """Write a minimal 64-byte dummy GGUF v3 header file and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 56)
    return target


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    text: str
    data: Any


def http_request(
    url: str,
    method: str = "GET",
    json_body: Any = None,
    raw_body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> HttpResponse:
    """Perform an HTTP request bypassing proxies and returning HttpResponse for 2xx/4xx/5xx."""
    req_headers: dict[str, str] = {}
    if headers:
        req_headers.update(headers)

    data_bytes: bytes | None = None
    if raw_body is not None:
        data_bytes = raw_body
    elif json_body is not None:
        data_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json; charset=utf-8")

    req = urllib.request.Request(
        url=url,
        data=data_bytes,
        headers=req_headers,
        method=method.upper(),
    )

    try:
        with _DIRECT_OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
            text = raw.decode("utf-8", errors="replace")
            parsed = _try_parse_json(text)
            return HttpResponse(
                status=int(resp.status),
                headers=resp_headers,
                text=text,
                data=parsed,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        resp_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        text = raw.decode("utf-8", errors="replace")
        parsed = _try_parse_json(text)
        return HttpResponse(
            status=int(exc.code),
            headers=resp_headers,
            text=text,
            data=parsed,
        )


def _try_parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class FakeLlamaServer:
    """Stdlib ThreadingHTTPServer double implementing llama-server /health and /v1/chat/completions."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        default_reply: str = "这是流萤的测试回复。",
        health_status: int = 200,
        health_sequence: list[int] | None = None,
        response_delay_sec: float = 0.0,
    ) -> None:
        self.host = host
        self._requested_port = port
        self.port = port
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop_event = threading.Event()

        self._default_reply = default_reply
        self._reply_fn: Callable[[Any, int], str] | None = None
        self._mode = "ok"
        self._error_status = 500
        self._response_delay_sec = response_delay_sec

        self._health_status = health_status
        self._health_sequence: deque[int] = deque(health_sequence or [])
        self._health_delay_sec = 0.0

        self._queued_responses: deque[dict[str, Any]] = deque()
        self._requests: list[dict[str, Any]] = []
        self._health_requests: list[dict[str, Any]] = []
        self._chat_counter = 0

        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._requests)

    @property
    def health_requests(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._health_requests)

    @property
    def last_request(self) -> dict[str, Any] | None:
        with self._lock:
            return self._requests[-1] if self._requests else None

    def set_reply(self, text: str) -> None:
        with self._lock:
            self._default_reply = text

    def set_reply_fn(self, fn: Callable[[Any, int], str] | None) -> None:
        with self._lock:
            self._reply_fn = fn

    def set_mode(
        self,
        mode: str,
        error_status: int = 500,
        delay_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._mode = mode
            self._error_status = error_status
            self._response_delay_sec = delay_sec

    def set_health(
        self,
        status: int = 200,
        sequence: list[int] | None = None,
        delay_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._health_status = status
            self._health_sequence = deque(sequence or [])
            self._health_delay_sec = delay_sec

    def enqueue_response(
        self,
        mode: str = "ok",
        reply: str | None = None,
        error_status: int = 500,
        delay_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._queued_responses.append(
                {
                    "mode": mode,
                    "reply": reply,
                    "error_status": error_status,
                    "delay_sec": delay_sec,
                }
            )

    def clear_requests(self) -> None:
        with self._lock:
            self._requests.clear()
            self._health_requests.clear()

    def wait_for_requests(self, count: int, timeout: float = 5.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._requests) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for {count} FakeLlamaServer requests; got {len(self._requests)}"
                    )
                self._cond.wait(timeout=remaining)
            return list(self._requests)

    def start(self) -> "FakeLlamaServer":
        if self._httpd is not None:
            return self
        self._stop_event.clear()
        parent = self

        class _Server(http.server.ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, status: int, payload: Any) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def _send_raw(self, status: int, raw: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                path_only = self.path.split("?", 1)[0]
                if path_only == "/health":
                    with parent._lock:
                        delay = parent._health_delay_sec
                        if parent._health_sequence:
                            status_code = parent._health_sequence.popleft()
                        else:
                            status_code = parent._health_status
                        parent._health_requests.append(
                            {
                                "timestamp": time.time(),
                                "path": self.path,
                                "headers": dict(self.headers.items()),
                                "status_returned": status_code,
                            }
                        )
                    if delay > 0 and parent._stop_event.wait(delay):
                        return
                    if status_code == 200:
                        self._send_json(200, {"status": "ok"})
                    else:
                        msg = "Loading model" if status_code == 503 else f"Simulated health {status_code}"
                        self._send_json(
                            status_code,
                            {"error": {"message": msg, "type": "unavailable_error", "code": status_code}},
                        )
                    return

                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                path_only = self.path.split("?", 1)[0]
                if path_only != "/v1/chat/completions":
                    self._send_json(404, {"error": "not found"})
                    return

                content_length = int(self.headers.get("Content-Length", "0") or "0")
                raw_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                raw_text = raw_bytes.decode("utf-8", errors="replace")
                try:
                    parsed_body = json.loads(raw_text) if raw_text.strip() else None
                except json.JSONDecodeError:
                    parsed_body = None

                with parent._cond:
                    parent._chat_counter += 1
                    call_idx = parent._chat_counter
                    parent._requests.append(
                        {
                            "timestamp": time.time(),
                            "path": self.path,
                            "headers": dict(self.headers.items()),
                            "body": parsed_body,
                            "raw_body": raw_text,
                        }
                    )
                    parent._cond.notify_all()

                    if parent._queued_responses:
                        item = parent._queued_responses.popleft()
                        mode = item["mode"]
                        explicit_reply = item["reply"]
                        err_status = item["error_status"]
                        delay = item["delay_sec"]
                    else:
                        mode = parent._mode
                        explicit_reply = None
                        err_status = parent._error_status
                        delay = parent._response_delay_sec

                    reply_fn = parent._reply_fn
                    default_reply = parent._default_reply

                if delay > 0 and parent._stop_event.wait(delay):
                    return

                if mode == "drop_connection":
                    try:
                        self.connection.close()
                    except OSError:
                        pass
                    return

                if mode == "malformed_json":
                    self._send_raw(200, b"{broken-json", content_type="application/json")
                    return

                if mode == "empty_choices":
                    self._send_json(
                        200,
                        {
                            "id": f"chatcmpl-empty-{call_idx}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": "fake-gguf",
                            "choices": [],
                        },
                    )
                    return

                if mode in ("http_500", "error"):
                    self._send_json(
                        err_status,
                        {
                            "error": {
                                "message": "Simulated llama-server error",
                                "type": "server_error",
                                "code": err_status,
                            }
                        },
                    )
                    return

                if mode in ("http_503", "loading"):
                    self._send_json(
                        503,
                        {
                            "error": {
                                "message": "Loading model",
                                "type": "unavailable_error",
                                "code": 503,
                            }
                        },
                    )
                    return

                if explicit_reply is not None:
                    reply_text = explicit_reply
                elif reply_fn is not None:
                    reply_text = str(reply_fn(parsed_body, call_idx))
                else:
                    reply_text = default_reply

                self._send_json(
                    200,
                    {
                        "id": f"chatcmpl-fake-{call_idx}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "fake-gguf",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": reply_text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": 16,
                            "total_tokens": 48,
                        },
                    },
                )

        self._httpd = _Server((self.host, self._requested_port), _Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "FakeLlamaServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


_MOCK_LLAMA_CS_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

class MockLlamaServer {
    private static readonly object LogLock = new object();
    private static readonly object HealthLock = new object();
    private static int healthCount = 0;
    private static int loadingUntil = 0;
    private static volatile bool running = true;

    static string EscapeJson(string s) {
        if (s == null) return "";
        var sb = new StringBuilder();
        foreach (char c in s) {
            if (c == '\\') sb.Append("\\\\");
            else if (c == '"') sb.Append("\\\"");
            else if (c == '\n') sb.Append("\\n");
            else if (c == '\r') sb.Append("\\r");
            else if (c == '\t') sb.Append("\\t");
            else sb.Append(c);
        }
        return sb.ToString();
    }

    static void LogArgs(string[] args) {
        string logPath = Environment.GetEnvironmentVariable("MOCK_LLAMA_ARGS_LOG");
        if (string.IsNullOrEmpty(logPath)) return;
        var sb = new StringBuilder();
        sb.Append("[");
        for (int i = 0; i < args.Length; i++) {
            if (i > 0) sb.Append(",");
            sb.Append("\"").Append(EscapeJson(args[i])).Append("\"");
        }
        sb.Append("]\n");
        lock (LogLock) {
            string dir = Path.GetDirectoryName(logPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            File.AppendAllText(logPath, sb.ToString(), new UTF8Encoding(false));
        }
    }

    static void SendHttp(NetworkStream stream, int code, string reason, string body) {
        byte[] bodyBytes = Encoding.UTF8.GetBytes(body);
        string header = string.Format(
            "HTTP/1.1 {0} {1}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {2}\r\nConnection: close\r\n\r\n",
            code, reason, bodyBytes.Length);
        byte[] headerBytes = Encoding.ASCII.GetBytes(header);
        stream.Write(headerBytes, 0, headerBytes.Length);
        stream.Write(bodyBytes, 0, bodyBytes.Length);
        stream.Flush();
    }

    static int FindHeaderEnd(byte[] data, int length) {
        for (int i = 0; i <= length - 4; i++) {
            if (data[i] == 13 && data[i + 1] == 10 && data[i + 2] == 13 && data[i + 3] == 10) {
                return i;
            }
        }
        return -1;
    }

    static void HandleClient(TcpClient client) {
        try {
            using (client)
            using (NetworkStream stream = client.GetStream()) {
                stream.ReadTimeout = 5000;
                byte[] buf = new byte[8192];
                int headerEnd = -1;
                byte[] initialBytes;
                int initialLen;
                using (MemoryStream headerStream = new MemoryStream()) {
                    while (true) {
                        int read = stream.Read(buf, 0, buf.Length);
                        if (read <= 0) break;
                        headerStream.Write(buf, 0, read);
                        initialBytes = headerStream.GetBuffer();
                        initialLen = (int)headerStream.Length;
                        headerEnd = FindHeaderEnd(initialBytes, initialLen);
                        if (headerEnd >= 0 || initialLen > 65536) break;
                    }
                    if (headerStream.Length == 0) return;
                    initialBytes = headerStream.ToArray();
                    initialLen = initialBytes.Length;
                }

                string headerBlock = headerEnd >= 0
                    ? Encoding.ASCII.GetString(initialBytes, 0, headerEnd)
                    : Encoding.ASCII.GetString(initialBytes, 0, initialLen);
                string[] lines = headerBlock.Split(new[] { "\r\n" }, StringSplitOptions.None);
                if (lines.Length == 0) return;
                string[] parts = lines[0].Split(' ');
                if (parts.Length < 2) return;
                string method = parts[0].ToUpperInvariant();
                string path = parts[1].Split('?')[0];

                int contentLength = 0;
                for (int i = 1; i < lines.Length; i++) {
                    int colon = lines[i].IndexOf(':');
                    if (colon > 0) {
                        string hName = lines[i].Substring(0, colon).Trim();
                        if (string.Equals(hName, "Content-Length", StringComparison.OrdinalIgnoreCase)) {
                            int.TryParse(lines[i].Substring(colon + 1).Trim(), out contentLength);
                        }
                    }
                }
                if (headerEnd >= 0 && contentLength > 0) {
                    using (MemoryStream bodyStream = new MemoryStream()) {
                        int bodyStart = headerEnd + 4;
                        int alreadyBody = initialLen - bodyStart;
                        if (alreadyBody > 0) {
                            bodyStream.Write(initialBytes, bodyStart, Math.Min(alreadyBody, contentLength));
                        }
                        byte[] tmp = new byte[4096];
                        while (bodyStream.Length < contentLength) {
                            int toRead = Math.Min(tmp.Length, contentLength - (int)bodyStream.Length);
                            int n = stream.Read(tmp, 0, toRead);
                            if (n <= 0) break;
                            bodyStream.Write(tmp, 0, n);
                        }
                        byte[] fullBody = bodyStream.ToArray();
                        string bodyText = fullBody.Length > 0 ? Encoding.UTF8.GetString(fullBody) : "";
                        if (bodyText.Length < 0) return;
                    }
                }

                if (method == "GET" && path == "/health") {
                    int count;
                    lock (HealthLock) {
                        healthCount++;
                        count = healthCount;
                    }
                    if (loadingUntil > 0 && count <= loadingUntil) {
                        SendHttp(stream, 503, "Service Unavailable",
                            "{\"error\":{\"code\":503,\"message\":\"Loading model\",\"type\":\"unavailable_error\"}}");
                    } else {
                        SendHttp(stream, 200, "OK", "{\"status\":\"ok\"}");
                    }
                } else if (method == "POST" && path == "/v1/chat/completions") {
                    SendHttp(stream, 200, "OK",
                        "{\"id\":\"chatcmpl-mock-exe\",\"object\":\"chat.completion\",\"created\":1700000000," +
                        "\"model\":\"mock-gguf\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"," +
                        "\"content\":\"Mock C# llama-server reply.\"},\"finish_reason\":\"stop\"}]," +
                        "\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}");
                } else {
                    SendHttp(stream, 404, "Not Found", "{\"error\":\"not found\"}");
                }
            }
        } catch {
            // Ignore closed client sockets during shutdown
        }
    }

    static int Main(string[] args) {
        LogArgs(args);

        string mode = Environment.GetEnvironmentVariable("MOCK_LLAMA_MODE") ?? "normal";
        if (string.Equals(mode, "crash", StringComparison.OrdinalIgnoreCase)) {
            Console.Error.WriteLine("Mock llama-server simulated crash");
            return 2;
        }
        if (mode.StartsWith("loading_then_ready:", StringComparison.OrdinalIgnoreCase)) {
            int.TryParse(mode.Substring("loading_then_ready:".Length), out loadingUntil);
        }

        string host = "127.0.0.1";
        int port = 8080;
        for (int i = 0; i < args.Length; i++) {
            if ((args[i] == "--host") && i + 1 < args.Length) {
                host = args[++i];
            } else if ((args[i] == "--port" || args[i] == "-p") && i + 1 < args.Length) {
                int.TryParse(args[++i], out port);
            }
        }

        Console.CancelKeyPress += delegate { running = false; };

        TcpListener listener = new TcpListener(IPAddress.Parse(host), port);
        listener.Start();
        try {
            while (running) {
                if (!listener.Pending()) {
                    Thread.Sleep(20);
                    continue;
                }
                TcpClient client = listener.AcceptTcpClient();
                Thread t = new Thread(() => HandleClient(client));
                t.IsBackground = true;
                t.Start();
            }
        } finally {
            listener.Stop();
        }
        return 0;
    }
}
"""


def _ensure_cached_mock_llama_exe() -> Path:
    global _MOCK_EXE_CACHE_DIR, _MOCK_EXE_CACHE_PATH
    with _MOCK_EXE_CACHE_LOCK:
        if _MOCK_EXE_CACHE_PATH is not None and _MOCK_EXE_CACHE_PATH.is_file():
            return _MOCK_EXE_CACHE_PATH
        if not CSC_EXE.is_file():
            raise FileNotFoundError(f"C# compiler not found at {CSC_EXE}")
        if _MOCK_EXE_CACHE_DIR is None or not _MOCK_EXE_CACHE_DIR.exists():
            _MOCK_EXE_CACHE_DIR = Path(tempfile.mkdtemp(prefix="moris_mock_llama_build_"))
        cs_path = _MOCK_EXE_CACHE_DIR / "mock_llama_server.cs"
        exe_path = _MOCK_EXE_CACHE_DIR / "llama-server.exe"
        cs_path.write_text(_MOCK_LLAMA_CS_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [str(CSC_EXE), "/nologo", "/optimize", f"/out:{exe_path}", str(cs_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode != 0 or not exe_path.is_file():
            raise RuntimeError(
                f"Failed to compile mock llama-server.exe (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}"
            )
        _MOCK_EXE_CACHE_PATH = exe_path
        return exe_path


def build_mock_llama_exe(output_dir: Path, exe_name: str = "llama-server.exe") -> Path:
    """Compile (cached) C# mock llama-server executable and place it in ``output_dir/exe_name``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _ensure_cached_mock_llama_exe()
    target = out_dir / exe_name
    shutil.copy2(cached, target)
    return target


def _ensure_cached_mock_python_exe(stdout_text: str, exit_code: int) -> Path:
    global _MOCK_EXE_CACHE_DIR
    key = (stdout_text, int(exit_code))
    with _MOCK_EXE_CACHE_LOCK:
        cached = _MOCK_PYTHON_EXE_CACHE.get(key)
        if cached is not None and cached.is_file():
            return cached
        if not CSC_EXE.is_file():
            raise FileNotFoundError(f"C# compiler not found at {CSC_EXE}")
        if _MOCK_EXE_CACHE_DIR is None or not _MOCK_EXE_CACHE_DIR.exists():
            _MOCK_EXE_CACHE_DIR = Path(tempfile.mkdtemp(prefix="moris_mock_llama_build_"))
        idx = len(_MOCK_PYTHON_EXE_CACHE)
        b64 = base64.b64encode(stdout_text.encode("utf-8")).decode("ascii")
        cs_source = f"""using System;
using System.Text;
class MockPython {{
    static int Main(string[] args) {{
        string msg = Encoding.UTF8.GetString(Convert.FromBase64String("{b64}"));
        if (msg.Length > 0) Console.WriteLine(msg);
        return {int(exit_code)};
    }}
}}
"""
        cs_path = _MOCK_EXE_CACHE_DIR / f"mock_python_{idx}.cs"
        exe_path = _MOCK_EXE_CACHE_DIR / f"mock_python_{idx}.exe"
        cs_path.write_text(cs_source, encoding="utf-8")
        proc = subprocess.run(
            [str(CSC_EXE), "/nologo", "/optimize", f"/out:{exe_path}", str(cs_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode != 0 or not exe_path.is_file():
            raise RuntimeError(
                f"Failed to compile mock python exe (rc={proc.returncode}): {proc.stdout}\n{proc.stderr}"
            )
        _MOCK_PYTHON_EXE_CACHE[key] = exe_path
        return exe_path


def build_mock_python_exe(
    output_dir: Path,
    exe_name: str,
    stdout_text: str,
    exit_code: int,
) -> Path:
    """Compile (cached) C# mock console executable printing ``stdout_text`` and exiting ``exit_code``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _ensure_cached_mock_python_exe(stdout_text, exit_code)
    target = out_dir / exe_name
    shutil.copy2(cached, target)
    return target


def read_mock_llama_args_log(log_path: Path) -> list[list[str]]:
    """Parse JSON-lines argv entries written by mock llama-server.exe."""
    if not log_path.is_file():
        return []
    entries: list[list[str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            entries.append(json.loads(stripped))
    return entries


def make_single_md_skill(
    directory: Path,
    filename: str,
    body: str,
    frontmatter: dict[str, str] | None = None,
    encoding: str = "utf-8",
    newline: str = "\n",
) -> Path:
    """Create a markdown skill file with optional YAML frontmatter."""
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    parts: list[str] = []
    if frontmatter:
        parts.append("---")
        for k, v in frontmatter.items():
            if "\n" in v:
                parts.append(f"{k}: |")
                for line in v.splitlines():
                    parts.append(f"  {line}")
            else:
                parts.append(f"{k}: {v}")
        parts.append("---")
        parts.append("")
    parts.append(body)
    content = "\n".join(parts)
    if newline != "\n":
        content = content.replace("\n", newline)
    target.write_bytes(content.encode(encoding))
    return target


def write_t2i_plugin(plugins_dir: Path, filename: str, source: str) -> Path:
    """Write a drop-in T2I plugin file into an isolated plugins directory."""
    p_dir = Path(plugins_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    target = p_dir / filename
    target.write_text(source, encoding="utf-8")
    return target


def make_recording_plugin_source(
    plugin_name: str,
    log_path: Path,
    use_register: bool = False,
    status: str = "stub_ok",
) -> str:
    """Generate a zero-backend-import drop-in T2I plugin recording SceneContext to JSONL."""
    log_path_repr = repr(str(log_path))
    plugin_name_repr = repr(plugin_name)
    status_repr = repr(status)
    export_block = (
        "def register(registry):\n"
        "    registry.register(_RecordingT2IPlugin())\n"
        if use_register
        else "PLUGIN = _RecordingT2IPlugin()\n"
    )
    return f'''from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class _PluginT2IResult:
    status: str
    plugin_name: str
    invoked: bool
    prompt_used: str = ""
    image_url: str | None = None
    image_path: str | None = None
    elapsed_ms: float = 0.0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class _RecordingT2IPlugin:
    name = {plugin_name_repr}
    description = "Opaque-box E2E recording stub plugin"

    def on_turn_completed(self, context, plugin_settings):
        recent = getattr(context, "recent_messages", ())
        entry = {{
            "session_id": getattr(context, "session_id", ""),
            "turn_index": getattr(context, "turn_index", 0),
            "timestamp_utc": getattr(context, "timestamp_utc", ""),
            "character_name": getattr(context, "character_name", ""),
            "skill_source": getattr(context, "skill_source", ""),
            "user_message": getattr(context, "user_message", ""),
            "assistant_reply": getattr(context, "assistant_reply", ""),
            "recent_messages": [dict(m) if isinstance(m, dict) else m for m in recent],
            "scene_prompt_hint": getattr(context, "scene_prompt_hint", ""),
            "metadata": dict(getattr(context, "metadata", {{}}) or {{}}),
            "plugin_settings": dict(plugin_settings or {{}}),
        }}
        log_file = Path({log_path_repr})
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\\n")

        prompt_used = f"{{entry['character_name']}}: {{entry['user_message']}} | {{entry['assistant_reply']}}"
        return _PluginT2IResult(
            status={status_repr},
            plugin_name=self.name,
            invoked=True,
            prompt_used=prompt_used,
            image_url=None,
            image_path=None,
            elapsed_ms=1.2,
            message="Recorded scene context",
            extra={{"recorded": True}},
        )


{export_block}'''


def make_crashing_plugin_source(
    plugin_name: str,
    error_message: str = "GPU OOM simulated",
) -> str:
    """Generate a drop-in T2I plugin that raises RuntimeError on turn completion."""
    return f'''class _CrashingT2IPlugin:
    name = {plugin_name!r}
    description = "Fault-injection crashing T2I plugin"

    def on_turn_completed(self, context, plugin_settings):
        raise RuntimeError({error_message!r})


PLUGIN = _CrashingT2IPlugin()
'''


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON objects from ``path``."""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            out.append(json.loads(s))
    return out


def find_browser_executable() -> Path | None:
    """Locate Microsoft Edge or Google Chrome executable for headless DOM verification."""
    for candidate in (EDGE_EXE, CHROME_EXE):
        if candidate.is_file():
            return candidate
    for name in ("msedge", "msedge.exe", "chrome", "chrome.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def run_headless_edge_dom(url: str, timeout: float = 20.0) -> tuple[int, str, str, Path]:
    """Run headless Edge/Chrome with --dump-dom against ``url`` and return (rc, stdout, stderr, exe)."""
    exe = find_browser_executable()
    if exe is None:
        raise FileNotFoundError("Neither msedge.exe nor chrome.exe was found on this system")
    user_data_dir = Path(tempfile.mkdtemp(prefix="moris_edge_dom_"))
    try:
        cmd = [
            str(exe),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={user_data_dir}",
            "--dump-dom",
            url,
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, exe
    finally:
        shutil.rmtree(user_data_dir, ignore_errors=True)


class AppProcessHarness:
    """Manages an isolated Moris Roleplay app.py or start.bat subprocess."""

    def __init__(
        self,
        repo_root: Path | None = None,
        config_overrides: dict[str, Any] | None = None,
        write_config: bool = True,
        config_path: Path | None = None,
        use_bat_launcher: bool = False,
        extra_cli_args: list[str] | None = None,
        isolated_plugins: bool = False,
        startup_timeout: float = 12.0,
        wait_for_health: bool = True,
        env_overrides: dict[str, str] | None = None,
        omit_no_browser: bool = False,
        port: int | None = None,
        host: str = "127.0.0.1",
        cwd: Path | None = None,
        pass_cli_port: bool = True,
        pass_cli_host: bool = True,
        pass_cli_config: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
        self.host = host
        self.port = port if port is not None else find_free_port(host)
        self.config_overrides = config_overrides or {}
        self.write_config = write_config
        self.use_bat_launcher = use_bat_launcher
        self.extra_cli_args = list(extra_cli_args or [])
        self.isolated_plugins = isolated_plugins
        self.startup_timeout = startup_timeout
        self.wait_for_health = wait_for_health
        self.env_overrides = dict(env_overrides or {})
        self.omit_no_browser = omit_no_browser
        self.cwd = Path(cwd) if cwd is not None else self.repo_root
        self.pass_cli_port = pass_cli_port
        self.pass_cli_host = pass_cli_host
        self.pass_cli_config = pass_cli_config

        self.temp_dir = Path(tempfile.mkdtemp(prefix="moris_e2e_"))
        self.config_path = Path(config_path) if config_path is not None else (self.temp_dir / "config.json")
        self.plugins_dir: Path | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_text: str = ""
        self.stderr_text: str = ""
        self._cleaned_up = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _preflight_target_check(self) -> None:
        app_py = self.repo_root / "app.py"
        if not app_py.is_file():
            raise FileNotFoundError(
                f"[E2E TARGET MISSING] {app_py} not found. Implementation Track M1-M4 must create it "
                "before running Tiers 1-4; run 'python -m unittest e2e_tests.test_infra_self -v' "
                "to verify E2E harness."
            )
        if self.use_bat_launcher:
            start_bat = self.repo_root / "start.bat"
            if not start_bat.is_file():
                raise FileNotFoundError(
                    f"[E2E TARGET MISSING] {start_bat} not found. Implementation Track M4 must create it "
                    "before running Tiers 1-4; run 'python -m unittest e2e_tests.test_infra_self -v' "
                    "to verify E2E harness."
                )

    def _prepare_files(self) -> None:
        overrides = deep_merge({}, self.config_overrides)
        if self.isolated_plugins:
            self.plugins_dir = self.temp_dir / "plugins" / "t2i"
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            t2i_section = overrides.setdefault("t2i", {})
            t2i_section.setdefault("plugins_dir", str(self.plugins_dir))

        if self.write_config:
            base_cfg: dict[str, Any] = {
                "server": {
                    "host": self.host,
                    "port": self.port,
                    "open_browser": False,
                }
            }
            merged = deep_merge(base_cfg, overrides)
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def start(self) -> "AppProcessHarness":
        if self.proc is not None:
            return self
        try:
            self._preflight_target_check()
            self._prepare_files()
        except Exception:
            self.cleanup_temp_dir()
            raise

        cli_args: list[str] = []
        if self.pass_cli_host:
            cli_args.extend(["--host", self.host])
        if self.pass_cli_port:
            cli_args.extend(["--port", str(self.port)])
        if self.pass_cli_config:
            cli_args.extend(["--config", str(self.config_path)])
        if not self.omit_no_browser:
            cli_args.append("--no-browser")
        cli_args.extend(self.extra_cli_args)

        cmd: str | list[str]
        if self.use_bat_launcher:
            if os.name == "nt":
                inner_cmd = subprocess.list2cmdline([str(self.repo_root / "start.bat"), *cli_args])
                cmd = f'cmd.exe /c "{inner_cmd}"'
            else:
                cmd = ["cmd.exe", "/c", str(self.repo_root / "start.bat"), *cli_args]
        else:
            cmd = [sys.executable, str(self.repo_root / "app.py"), *cli_args]

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.update(self.env_overrides)

        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE if self.use_bat_launcher else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )

        if self.wait_for_health:
            self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_err = ""
        while time.monotonic() < deadline:
            assert self.proc is not None
            rc = self.proc.poll()
            if rc is not None:
                self._drain_pipes()
                raise RuntimeError(
                    f"App process exited prematurely with code {rc}.\n"
                    f"STDOUT:\n{self.stdout_text}\nSTDERR:\n{self.stderr_text}"
                )
            try:
                resp = self.get_json("/api/health", timeout=1.5)
                if resp.status == 200:
                    return
                last_err = f"HTTP {resp.status}: {resp.text}"
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(0.1)

        self.stop(cleanup_temp=False)
        raise TimeoutError(
            f"App server did not become healthy on {self.base_url} within {self.startup_timeout}s. "
            f"Last error: {last_err}\nSTDOUT:\n{self.stdout_text}\nSTDERR:\n{self.stderr_text}"
        )

    def _drain_pipes(self, timeout: float = 2.0) -> None:
        if self.proc is None:
            return
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

        pipes_to_drain: list[tuple[str, Any, list[str]]] = []
        if self.proc.stdout is not None:
            pipes_to_drain.append(("stdout", self.proc.stdout, []))
        if self.proc.stderr is not None:
            pipes_to_drain.append(("stderr", self.proc.stderr, []))

        threads: list[tuple[str, Any, list[str], threading.Thread]] = []
        for label, pipe, bucket in pipes_to_drain:
            def _reader(p: Any = pipe, out: list[str] = bucket) -> None:
                try:
                    out.append(p.read())
                except Exception:
                    pass

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            threads.append((label, pipe, bucket, t))

        deadline = time.monotonic() + timeout
        for label, pipe, bucket, t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
            if bucket:
                text_val = "".join(bucket)
                if label == "stdout":
                    self.stdout_text += text_val
                else:
                    self.stderr_text += text_val
            if not t.is_alive():
                try:
                    pipe.close()
                except Exception:
                    pass

    def send_ctrl_break(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if os.name == "nt":
            self.proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            self.proc.send_signal(signal.SIGINT)

    def stop(self, cleanup_temp: bool = True) -> None:
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        try:
                            self.proc.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            self.proc.terminate()
                            try:
                                self.proc.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                self.proc.kill()
                                try:
                                    self.proc.wait(timeout=2.0)
                                except subprocess.TimeoutExpired:
                                    pass
                    else:
                        self.proc.terminate()
                        try:
                            self.proc.wait(timeout=4.0)
                        except subprocess.TimeoutExpired:
                            self.proc.kill()
                            try:
                                self.proc.wait(timeout=3.0)
                            except subprocess.TimeoutExpired:
                                pass
            finally:
                self._drain_pipes(timeout=2.0)
                self.proc = None

        if cleanup_temp:
            self.cleanup_temp_dir()

    def cleanup_temp_dir(self) -> None:
        if not self._cleaned_up and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self._cleaned_up = True

    def request(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> HttpResponse:
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.base_url}{path}"
        return http_request(
            url=url,
            method=method,
            json_body=json_body,
            raw_body=raw_body,
            headers=headers,
            timeout=timeout,
        )

    def get_json(self, path: str, timeout: float = 15.0) -> HttpResponse:
        return self.request("GET", path, timeout=timeout)

    def post_json(self, path: str, payload: Any = None, timeout: float = 15.0) -> HttpResponse:
        return self.request("POST", path, json_body=payload, timeout=timeout)

    def put_json(self, path: str, payload: Any = None, timeout: float = 15.0) -> HttpResponse:
        return self.request("PUT", path, json_body=payload, timeout=timeout)

    def get_text(self, path: str, timeout: float = 15.0) -> HttpResponse:
        return self.request("GET", path, timeout=timeout)

    def __enter__(self) -> "AppProcessHarness":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop(cleanup_temp=True)
