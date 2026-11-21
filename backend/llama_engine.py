"""Pure-stdlib llama.cpp engine adapter for managed server, external server, and CLI modes."""

from __future__ import annotations

import atexit
import dataclasses
import http.client
import json
import math
import os
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

EngineState = Literal["stopped", "starting", "ready", "error"]

ERROR_HTTP_STATUS: dict[str, int] = {
    "CONFIG_INCOMPLETE": 400,
    "ENGINE_BINARY_NOT_FOUND": 400,
    "MODEL_FILE_INVALID": 400,
    "ENGINE_PORT_CONFLICT": 409,
    "ENGINE_PROCESS_EXITED": 502,
    "ENGINE_LOADING": 503,
    "ENGINE_TIMEOUT": 504,
    "ENGINE_UPSTREAM_ERROR": 502,
}

if os.name == "nt":
    _SERVER_BIN_CANDIDATES: tuple[str, ...] = (
        "llama-server.exe",
        "server.exe",
        "llama-server.cmd",
        "llama-server.bat",
        "server.cmd",
        "server.bat",
    )
    _CLI_BIN_CANDIDATES: tuple[str, ...] = (
        "llama-cli.exe",
        "main.exe",
        "llama-cli.cmd",
        "llama-cli.bat",
        "main.cmd",
        "main.bat",
    )
    _ALLOWED_WIN_SUFFIXES: frozenset[str] = frozenset({".exe", ".cmd", ".bat", ".com"})
else:
    _SERVER_BIN_CANDIDATES = (
        "llama-server",
        "server",
        "llama-server.sh",
        "server.sh",
    )
    _CLI_BIN_CANDIDATES = (
        "llama-cli",
        "main",
        "llama-cli.sh",
        "main.sh",
    )
    _ALLOWED_WIN_SUFFIXES = frozenset()

_STDERR_RING_MAXLINES = 200
_CLI_INLINE_PROMPT_LIMIT = 8000
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

_MANAGED_ENGINES: weakref.WeakSet[LlamaServerEngine] = weakref.WeakSet()
_MANAGED_ENGINES_LOCK = threading.Lock()


def _shutdown_all_managed_engines() -> None:
    with _MANAGED_ENGINES_LOCK:
        engines = list(_MANAGED_ENGINES)
    for engine in engines:
        try:
            engine.shutdown()
        except Exception:
            pass


atexit.register(_shutdown_all_managed_engines)


class EngineError(Exception):
    """Structured error raised by llama.cpp engines for non-crashing API responses."""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int | str | None = None,
        detail: str = "",
    ) -> None:
        if http_status is None:
            resolved_status = ERROR_HTTP_STATUS.get(code, 502)
        elif isinstance(http_status, str):
            if not detail:
                detail = http_status
            resolved_status = ERROR_HTTP_STATUS.get(code, 502)
        else:
            resolved_status = int(http_status)
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = resolved_status
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }

    def to_error_payload(self) -> dict[str, dict[str, str]]:
        return {"error": self.to_dict()}


@dataclass(slots=True)
class EngineStatus:
    """Snapshot of the llama.cpp engine lifecycle state."""

    state: EngineState = "stopped"
    mode: str = ""
    endpoint: str = ""
    pid: int | None = None
    error_code: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LlamaEngine(ABC):
    """Abstract base class for llama.cpp engine adapters."""

    @abstractmethod
    def status(self) -> EngineStatus:
        """Return a defensive copy of the current engine status."""

    @abstractmethod
    def ensure_ready(self) -> None:
        """Validate configuration and ensure the engine is ready to serve requests."""

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a chat completion request and return a normalized response dict."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release any managed process or runtime resources idempotently."""


def _is_loopback_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    return host in _LOOPBACK_HOSTS


def _open_url(request: urllib.request.Request, timeout: float) -> Any:
    if _is_loopback_url(request.full_url):
        return _DIRECT_OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def _decode_bytes(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def _is_loading_health_payload(raw_body: bytes) -> bool:
    if not raw_body:
        return False
    try:
        parsed = json.loads(raw_body.decode("utf-8", errors="replace"))
    except ValueError:
        return False
    if isinstance(parsed, dict):
        status_val = str(parsed.get("status", "")).strip().lower()
        return status_val.startswith("loading")
    return False


def resolve_executable(raw_path: str, candidates: tuple[str, ...]) -> Path:
    """Resolve a llama.cpp binary from a file path or installation directory."""
    cleaned = raw_path.strip()
    if not cleaned:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "llama.cpp path is not configured.",
            http_status=400,
        )

    path = Path(cleaned).expanduser()
    if not path.exists():
        raise EngineError(
            "ENGINE_BINARY_NOT_FOUND",
            "llama.cpp executable or directory was not found.",
            http_status=400,
            detail=str(path),
        )

    if path.is_file():
        if os.name == "nt":
            if path.suffix.lower() not in _ALLOWED_WIN_SUFFIXES:
                raise EngineError(
                    "ENGINE_BINARY_NOT_FOUND",
                    "Configured llama.cpp file is not an executable Windows binary.",
                    http_status=400,
                    detail=str(path),
                )
        elif not os.access(path, os.X_OK):
            raise EngineError(
                "ENGINE_BINARY_NOT_FOUND",
                "Configured llama.cpp binary is not executable.",
                http_status=400,
                detail=str(path),
            )
        return path.resolve()

    if path.is_dir():
        for name in candidates:
            candidate = path / name
            if candidate.is_file():
                if os.name == "nt" or os.access(candidate, os.X_OK):
                    return candidate.resolve()
        raise EngineError(
            "ENGINE_BINARY_NOT_FOUND",
            "No supported llama.cpp executable found in the configured directory.",
            http_status=400,
            detail=f"{path} (searched: {', '.join(candidates)})",
        )

    raise EngineError(
        "ENGINE_BINARY_NOT_FOUND",
        "Configured llama.cpp path is neither a regular file nor a directory.",
        http_status=400,
        detail=str(path),
    )


def validate_gguf_model(raw_path: str) -> Path:
    """Validate that the configured model path points to a readable non-empty .gguf file."""
    cleaned = raw_path.strip()
    if not cleaned:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "GGUF model path is not configured.",
            http_status=400,
        )

    path = Path(cleaned).expanduser()
    if not path.exists():
        raise EngineError(
            "MODEL_FILE_INVALID",
            "GGUF model file does not exist.",
            http_status=400,
            detail=str(path),
        )
    if not path.is_file():
        raise EngineError(
            "MODEL_FILE_INVALID",
            "GGUF model path must point to a file, not a directory.",
            http_status=400,
            detail=str(path),
        )
    if path.suffix.lower() != ".gguf":
        raise EngineError(
            "MODEL_FILE_INVALID",
            "Model file must have a .gguf extension.",
            http_status=400,
            detail=str(path),
        )

    try:
        size = path.stat().st_size
        if size <= 0:
            raise EngineError(
                "MODEL_FILE_INVALID",
                "GGUF model file is empty (0 bytes).",
                http_status=400,
                detail=str(path),
            )
        with path.open("rb") as handle:
            first_byte = handle.read(1)
        if not first_byte:
            raise EngineError(
                "MODEL_FILE_INVALID",
                "GGUF model file could not be read.",
                http_status=400,
                detail=str(path),
            )
    except EngineError:
        raise
    except OSError as exc:
        raise EngineError(
            "MODEL_FILE_INVALID",
            "Failed to read GGUF model file.",
            http_status=400,
            detail=f"{path}: {exc}",
        ) from exc

    return path.resolve()


def probe_tcp_port_available(host: str, port: int) -> None:
    """Verify the target TCP host/port is not already bound before starting llama-server."""
    raw_host = host.strip()
    if raw_host.startswith("[") and raw_host.endswith("]"):
        raw_host = raw_host[1:-1]
    bind_host = "127.0.0.1" if raw_host in ("", "localhost") else raw_host
    if raw_host in ("", "0.0.0.0", "localhost"):
        connect_host = "127.0.0.1"
    elif raw_host == "::":
        connect_host = "::1"
    else:
        connect_host = raw_host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET

    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe_sock:
            probe_sock.settimeout(0.15)
            if probe_sock.connect_ex((connect_host, port)) == 0:
                raise EngineError(
                    "ENGINE_PORT_CONFLICT",
                    f"Port {port} on {bind_host} is already in use.",
                    http_status=409,
                    detail=f"Active listener detected on {connect_host}:{port}",
                )
    except EngineError:
        raise
    except OSError:
        pass

    try:
        with socket.socket(family, socket.SOCK_STREAM) as bind_sock:
            if os.name == "nt":
                exclusive_opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive_opt is not None:
                    bind_sock.setsockopt(socket.SOL_SOCKET, exclusive_opt, 1)
            else:
                bind_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bind_sock.bind((bind_host, port))
        if os.name == "nt" and family == socket.AF_INET and bind_host == "127.0.0.1":
            with socket.socket(family, socket.SOCK_STREAM) as wildcard_sock:
                exclusive_opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
                if exclusive_opt is not None:
                    wildcard_sock.setsockopt(socket.SOL_SOCKET, exclusive_opt, 1)
                wildcard_sock.bind(("0.0.0.0", port))
    except OSError as exc:
        raise EngineError(
            "ENGINE_PORT_CONFLICT",
            f"Port {port} on {bind_host} is already in use.",
            http_status=409,
            detail=str(exc),
        ) from exc


def _validate_sampling_params(
    params: Any,
    *,
    default_temperature: float,
    default_top_p: float,
    default_max_tokens: int,
) -> tuple[float, float, int]:
    if params is None:
        p: Mapping[str, Any] = {}
    elif not isinstance(params, Mapping):
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "Optional 'params' must be a mapping.",
            http_status=400,
        )
    else:
        p = params

    raw_temp = p.get("temperature", default_temperature)
    if (
        isinstance(raw_temp, bool)
        or not isinstance(raw_temp, (int, float))
        or not math.isfinite(float(raw_temp))
    ):
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "Parameter 'temperature' must be a finite number.",
            http_status=400,
        )

    raw_top_p = p.get("top_p", default_top_p)
    if (
        isinstance(raw_top_p, bool)
        or not isinstance(raw_top_p, (int, float))
        or not math.isfinite(float(raw_top_p))
    ):
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "Parameter 'top_p' must be a finite number.",
            http_status=400,
        )

    raw_max_tokens = p.get("max_tokens", default_max_tokens)
    if isinstance(raw_max_tokens, bool) or not isinstance(raw_max_tokens, int) or raw_max_tokens < 1:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "Parameter 'max_tokens' must be an integer >= 1.",
            http_status=400,
        )

    return float(raw_temp), float(raw_top_p), int(raw_max_tokens)


def _terminate_proc_resources(
    proc: subprocess.Popen[str] | None,
    exe: Path | None,
    thread: threading.Thread | None,
) -> None:
    if proc is not None:
        if proc.poll() is None:
            is_batch = os.name == "nt" and exe is not None and exe.suffix.lower() in (".bat", ".cmd")
            if is_batch:
                try:
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5,
                        creationflags=_CREATE_NO_WINDOW,
                        check=False,
                    )
                except Exception:
                    pass
            else:
                try:
                    proc.terminate()
                except Exception:
                    pass

            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=5,
                            creationflags=_CREATE_NO_WINDOW,
                            check=False,
                        )
                    except Exception:
                        pass
                else:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass

        if proc.stderr is not None and not proc.stderr.closed:
            try:
                proc.stderr.close()
            except Exception:
                pass

    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)


def _finalize_proc_box(proc_box: dict[str, Any]) -> None:
    proc = proc_box.get("proc")
    exe = proc_box.get("exe")
    thread = proc_box.get("thread")
    proc_box["proc"] = None
    proc_box["exe"] = None
    proc_box["thread"] = None
    _terminate_proc_resources(proc, exe, thread)


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "messages must be a non-empty list of role/content dicts.",
            http_status=400,
        )
    normalized: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, Mapping) or "role" not in item or "content" not in item:
            raise EngineError(
                "CONFIG_INCOMPLETE",
                "Each message must be a dict containing 'role' and 'content'.",
                http_status=400,
            )
        raw_content = item["content"]
        if isinstance(raw_content, list):
            blocks: list[dict[str, Any]] = []
            for block in raw_content:
                if not isinstance(block, Mapping) or "type" not in block:
                    raise EngineError(
                        "CONFIG_INCOMPLETE",
                        "Each multimodal content block must be a dict with a 'type' key.",
                        http_status=400,
                    )
                blocks.append(dict(block))
            content_val: str | list[dict[str, Any]] = blocks
        else:
            content_val = str(raw_content)

        normalized.append(
            {
                "role": str(item["role"]),
                "content": content_val,
            }
        )
    return normalized


def _validate_http_url(raw_url: str) -> str:
    cleaned = raw_url.strip().rstrip("/")
    if not cleaned:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "External llama.cpp server_url is not configured.",
            http_status=400,
        )
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "External llama.cpp server_url must be a valid http:// or https:// URL.",
            http_status=400,
            detail=cleaned,
        )
    return cleaned


class LlamaServerEngine(LlamaEngine):
    """Adapter for llama-server in managed_server or external_server mode."""

    def __init__(self, llama_config: Mapping[str, Any]) -> None:
        cfg = dict(llama_config)
        self.mode: str = str(cfg.get("engine_mode", "managed_server")).strip() or "managed_server"
        if self.mode not in ("managed_server", "external_server"):
            raise EngineError(
                "CONFIG_INCOMPLETE",
                f"Unsupported server engine_mode: {self.mode!r}",
                http_status=400,
            )

        self.llama_cpp_path: str = str(cfg.get("llama_cpp_path", ""))
        self.model_path: str = str(cfg.get("model_path", ""))
        self.mmproj_path: str = str(cfg.get("mmproj_path", "") or "")
        self.multimodal_enabled: bool = bool(cfg.get("multimodal_enabled", True))
        self.server_url: str = str(cfg.get("server_url", "http://127.0.0.1:8080"))
        self.host: str = str(cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
        self.port: int = int(cfg.get("port", 8080))
        self.ctx_size: int = int(cfg.get("ctx_size", 16384))
        self.n_gpu_layers: int = int(cfg.get("n_gpu_layers", -1))
        self.startup_timeout_sec: float = float(cfg.get("startup_timeout_sec", 60))
        self.request_timeout_sec: float = float(cfg.get("request_timeout_sec", 120))
        self.temperature: float = float(cfg.get("temperature", 0.8))
        self.top_p: float = float(cfg.get("top_p", 0.9))
        self.max_tokens: int = int(cfg.get("max_tokens", 512))

        if self.mode == "managed_server":
            if self.host in ("", "0.0.0.0"):
                connect_host = "127.0.0.1"
            elif self.host == "::":
                connect_host = "::1"
            else:
                connect_host = self.host
            url_host = (
                f"[{connect_host}]"
                if ":" in connect_host and not connect_host.startswith("[")
                else connect_host
            )
            self.endpoint: str = f"http://{url_host}:{self.port}"
        else:
            self.endpoint = self.server_url.strip().rstrip("/")

        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._resolved_exe: Path | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proc_box: dict[str, Any] = {"proc": None, "exe": None, "thread": None}
        self._stderr_lines: deque[str] = deque(maxlen=_STDERR_RING_MAXLINES)
        self._status = EngineStatus(
            state="stopped",
            mode=self.mode,
            endpoint=self.endpoint,
            pid=None,
            error_code="",
            detail="",
        )

        if self.mode == "managed_server":
            self._finalizer = weakref.finalize(self, _finalize_proc_box, self._proc_box)
            with _MANAGED_ENGINES_LOCK:
                _MANAGED_ENGINES.add(self)
        else:
            self._finalizer = None

    def _set_status(
        self,
        *,
        state: EngineState,
        pid: int | None = None,
        error_code: str = "",
        detail: str = "",
    ) -> None:
        with self._state_lock:
            self._status = EngineStatus(
                state=state,
                mode=self.mode,
                endpoint=self.endpoint,
                pid=pid,
                error_code=error_code,
                detail=detail,
            )

    def _stderr_tail(self, limit: int = 20) -> str:
        with self._state_lock:
            lines = list(self._stderr_lines)[-limit:]
        return "\n".join(lines).strip()

    @staticmethod
    def _start_stderr_reader(
        pipe: Any,
        state_lock: threading.Lock,
        stderr_lines: deque[str],
    ) -> threading.Thread:
        def _reader() -> None:
            try:
                for line in iter(pipe.readline, ""):
                    cleaned = line.rstrip("\r\n")
                    with state_lock:
                        stderr_lines.append(cleaned)
            except Exception:
                pass
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_reader, name="llama-server-stderr", daemon=True)
        thread.start()
        return thread

    def _terminate_proc_locked(self) -> None:
        with self._state_lock:
            proc = self._proc
            exe = self._resolved_exe
            thread = self._stderr_thread
            self._proc = None
            self._resolved_exe = None
            self._stderr_thread = None
            self._proc_box["proc"] = None
            self._proc_box["exe"] = None
            self._proc_box["thread"] = None

        _terminate_proc_resources(proc, exe, thread)

    def _probe_health(self, timeout: float) -> tuple[str, str]:
        """Return (kind, detail) where kind in {'ok', 'loading', 'timeout', 'error'}."""
        req = urllib.request.Request(f"{self.endpoint}/health", method="GET")
        try:
            with _open_url(req, timeout=timeout) as resp:
                status_code = getattr(resp, "status", 200)
                raw_body = resp.read()
                if status_code == 200:
                    if _is_loading_health_payload(raw_body):
                        return "loading", _decode_bytes(raw_body) or "Model is still loading."
                    return "ok", ""
                if status_code == 503:
                    return "loading", _decode_bytes(raw_body) or "HTTP 503 Service Unavailable"
                return "error", f"HTTP {status_code}: {_decode_bytes(raw_body)}"
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = _decode_bytes(exc.read())
            except Exception:
                body_text = str(exc.reason or "")
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
            if exc.code == 503:
                return "loading", body_text or "HTTP 503 Service Unavailable"
            return "error", f"HTTP {exc.code}: {body_text}".strip(": ")
        except TimeoutError as exc:
            return "timeout", str(exc) or "Health probe timed out."
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return "timeout", str(exc.reason) or "Health probe timed out."
            return "error", str(exc.reason or exc)
        except (http.client.HTTPException, OSError) as exc:
            return "error", str(exc)

    def status(self) -> EngineStatus:
        with self._state_lock:
            return dataclasses.replace(self._status)

    def ensure_ready(self) -> None:
        with self._lifecycle_lock:
            if self.mode == "external_server":
                try:
                    endpoint = _validate_http_url(self.server_url)
                except EngineError as exc:
                    self._set_status(
                        state="error",
                        pid=None,
                        error_code=exc.code,
                        detail=exc.detail or exc.message,
                    )
                    raise

                self.endpoint = endpoint
                self._set_status(state="starting", pid=None)
                kind, detail = self._probe_health(
                    timeout=min(max(0.05, float(self.startup_timeout_sec)), 5.0)
                )
                if kind == "ok":
                    self._set_status(state="ready", pid=None)
                    return
                if kind == "loading":
                    err = EngineError(
                        "ENGINE_LOADING",
                        "External llama.cpp server is still loading the model.",
                        http_status=503,
                        detail=detail,
                    )
                    self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                    raise err
                if kind == "timeout":
                    err = EngineError(
                        "ENGINE_TIMEOUT",
                        "Timed out waiting for external llama.cpp server health check.",
                        http_status=504,
                        detail=detail,
                    )
                    self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                    raise err
                err = EngineError(
                    "ENGINE_UPSTREAM_ERROR",
                    "External llama.cpp server health check failed.",
                    http_status=502,
                    detail=detail,
                )
                self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                raise err

            # managed_server mode
            existing_proc = self._proc
            if existing_proc is not None and existing_proc.poll() is None:
                kind, _ = self._probe_health(timeout=1.5)
                if kind == "ok":
                    self._set_status(state="ready", pid=existing_proc.pid)
                    return

            if existing_proc is not None:
                self._terminate_proc_locked()

            try:
                if not self.llama_cpp_path.strip() or not self.model_path.strip():
                    raise EngineError(
                        "CONFIG_INCOMPLETE",
                        "Both llama_cpp_path and model_path are required in managed_server mode.",
                        http_status=400,
                    )
                exe = resolve_executable(self.llama_cpp_path, _SERVER_BIN_CANDIDATES)
                model = validate_gguf_model(self.model_path)
                mmproj: Path | None = (
                    validate_gguf_model(self.mmproj_path)
                    if self.mmproj_path.strip()
                    else None
                )
                probe_tcp_port_available(self.host, self.port)
            except EngineError as exc:
                self._set_status(
                    state="error",
                    pid=None,
                    error_code=exc.code,
                    detail=exc.detail or exc.message,
                )
                raise

            cmd = [
                str(exe),
                "-m",
                str(model),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "-c",
                str(self.ctx_size),
                "-ngl",
                str(self.n_gpu_layers),
            ]
            if mmproj is not None:
                cmd.extend(["--mmproj", str(mmproj)])
            with self._state_lock:
                self._stderr_lines.clear()

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=_CREATE_NO_WINDOW,
                )
            except (FileNotFoundError, PermissionError, OSError) as exc:
                err = EngineError(
                    "ENGINE_BINARY_NOT_FOUND",
                    "Failed to launch llama-server executable.",
                    http_status=400,
                    detail=f"{exe}: {exc}",
                )
                self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                raise err from exc

            stderr_thread = self._start_stderr_reader(
                proc.stderr,
                self._state_lock,
                self._stderr_lines,
            )
            with self._state_lock:
                self._proc = proc
                self._resolved_exe = exe
                self._stderr_thread = stderr_thread
                self._proc_box["proc"] = proc
                self._proc_box["exe"] = exe
                self._proc_box["thread"] = stderr_thread
                self._status = EngineStatus(
                    state="starting",
                    mode=self.mode,
                    endpoint=self.endpoint,
                    pid=proc.pid,
                    error_code="",
                    detail="",
                )

            deadline = time.monotonic() + max(0.05, self.startup_timeout_sec)
            last_was_503 = False
            last_detail = ""

            while True:
                rc = proc.poll()
                if rc is not None:
                    if self._stderr_thread is not None:
                        self._stderr_thread.join(timeout=0.5)
                    stderr_tail = self._stderr_tail()
                    self._terminate_proc_locked()
                    detail = f"exit_code={rc}" + (f"; stderr: {stderr_tail}" if stderr_tail else "")
                    err = EngineError(
                        "ENGINE_PROCESS_EXITED",
                        f"llama-server exited prematurely with code {rc}.",
                        http_status=502,
                        detail=detail,
                    )
                    self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                    raise err

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                probe_timeout = min(1.5, max(0.2, remaining))
                kind, detail = self._probe_health(timeout=probe_timeout)
                if kind == "ok":
                    self._set_status(state="ready", pid=proc.pid)
                    return
                if kind == "loading":
                    last_was_503 = True
                    last_detail = detail
                else:
                    last_was_503 = False
                    last_detail = detail

                rc = proc.poll()
                if rc is not None:
                    continue

                sleep_for = min(0.1, max(0.0, deadline - time.monotonic()))
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Final check if process exited right as deadline expired
            rc = proc.poll()
            if rc is not None:
                if self._stderr_thread is not None:
                    self._stderr_thread.join(timeout=0.5)
                stderr_tail = self._stderr_tail()
                self._terminate_proc_locked()
                detail = f"exit_code={rc}" + (f"; stderr: {stderr_tail}" if stderr_tail else "")
                err = EngineError(
                    "ENGINE_PROCESS_EXITED",
                    f"llama-server exited prematurely with code {rc}.",
                    http_status=502,
                    detail=detail,
                )
                self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
                raise err

            stderr_tail = self._stderr_tail()
            self._terminate_proc_locked()
            combined_detail = last_detail
            if stderr_tail:
                combined_detail = f"{last_detail}; stderr: {stderr_tail}" if last_detail else stderr_tail

            if last_was_503:
                err = EngineError(
                    "ENGINE_LOADING",
                    "llama-server did not finish loading the model within startup_timeout_sec.",
                    http_status=503,
                    detail=combined_detail,
                )
            else:
                err = EngineError(
                    "ENGINE_TIMEOUT",
                    "Timed out waiting for llama-server /health to become ready.",
                    http_status=504,
                    detail=combined_detail,
                )
            self._set_status(state="error", pid=None, error_code=err.code, detail=err.detail)
            raise err

    def _finalize_chat_error(
        self,
        active_proc: subprocess.Popen[str] | None,
        current_pid: int | None,
        err: EngineError,
    ) -> EngineError:
        with self._state_lock:
            if self.mode == "managed_server" and (
                self._proc is not active_proc or self._status.state == "stopped"
            ):
                return EngineError(
                    "ENGINE_PROCESS_EXITED",
                    "llama-server stopped while completion request was in flight.",
                    http_status=502,
                    detail=err.detail,
                )
            self._status = EngineStatus(
                state="error",
                mode=self.mode,
                endpoint=self.endpoint,
                pid=current_pid,
                error_code=err.code,
                detail=err.detail,
            )
            return err

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_messages = _validate_messages(messages)
        temperature, top_p, max_tokens = _validate_sampling_params(
            params,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
            default_max_tokens=self.max_tokens,
        )

        active_proc: subprocess.Popen[str] | None = None
        if self.mode == "managed_server":
            with self._lifecycle_lock:
                proc = self._proc
                if proc is None:
                    needs_start = True
                    exited_rc = None
                else:
                    exited_rc = proc.poll()
                    needs_start = False
                    if exited_rc is not None:
                        if self._stderr_thread is not None:
                            self._stderr_thread.join(timeout=0.5)
                        stderr_tail = self._stderr_tail()
                        self._terminate_proc_locked()
                        detail = f"exit_code={exited_rc}" + (
                            f"; stderr: {stderr_tail}" if stderr_tail else ""
                        )
                        err = EngineError(
                            "ENGINE_PROCESS_EXITED",
                            f"llama-server exited unexpectedly with code {exited_rc}.",
                            http_status=502,
                            detail=detail,
                        )
                        self._set_status(
                            state="error",
                            pid=None,
                            error_code=err.code,
                            detail=err.detail,
                        )
                        raise err
                    active_proc = proc

            if needs_start:
                self.ensure_ready()
                with self._state_lock:
                    active_proc = self._proc
                    if active_proc is None or self._status.state == "stopped":
                        raise EngineError(
                            "ENGINE_PROCESS_EXITED",
                            "llama-server was stopped.",
                            http_status=502,
                        )
        else:
            try:
                self.endpoint = _validate_http_url(self.server_url)
            except EngineError as exc:
                self._set_status(
                    state="error",
                    pid=None,
                    error_code=exc.code,
                    detail=exc.detail or exc.message,
                )
                raise

        payload = {
            "messages": normalized_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body_bytes = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=body_bytes,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        with self._state_lock:
            if self.mode == "managed_server" and (
                active_proc is None or self._proc is not active_proc or self._status.state == "stopped"
            ):
                raise EngineError(
                    "ENGINE_PROCESS_EXITED",
                    "llama-server was stopped.",
                    http_status=502,
                )
            current_pid = (
                active_proc.pid
                if active_proc is not None and active_proc.poll() is None
                else None
            )

        request_timeout = max(0.05, float(self.request_timeout_sec))
        try:
            with _open_url(req, timeout=request_timeout) as resp:
                raw_resp = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = _decode_bytes(exc.read())
            except Exception:
                err_body = str(exc.reason or "")
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
            if exc.code == 503:
                err = EngineError(
                    "ENGINE_LOADING",
                    "llama-server is currently loading the model.",
                    http_status=503,
                    detail=err_body or "HTTP 503 Service Unavailable",
                )
            else:
                err = EngineError(
                    "ENGINE_UPSTREAM_ERROR",
                    f"llama-server returned HTTP {exc.code}.",
                    http_status=502,
                    detail=err_body or str(exc.reason or ""),
                )
            raise self._finalize_chat_error(active_proc, current_pid, err) from exc
        except TimeoutError as exc:
            err = EngineError(
                "ENGINE_TIMEOUT",
                "Timed out waiting for llama-server completion response.",
                http_status=504,
                detail=str(exc),
            )
            raise self._finalize_chat_error(active_proc, current_pid, err) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                err = EngineError(
                    "ENGINE_TIMEOUT",
                    "Timed out waiting for llama-server completion response.",
                    http_status=504,
                    detail=str(exc.reason),
                )
            else:
                err = EngineError(
                    "ENGINE_UPSTREAM_ERROR",
                    "Failed to connect to llama-server.",
                    http_status=502,
                    detail=str(exc.reason or exc),
                )
            raise self._finalize_chat_error(active_proc, current_pid, err) from exc
        except (http.client.HTTPException, OSError) as exc:
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "Connection error while communicating with llama-server.",
                http_status=502,
                detail=str(exc),
            )
            raise self._finalize_chat_error(active_proc, current_pid, err) from exc

        try:
            data = json.loads(raw_resp.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "llama-server returned invalid JSON.",
                http_status=502,
                detail=str(exc),
            )
            raise self._finalize_chat_error(active_proc, current_pid, err) from exc

        if not isinstance(data, dict):
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "llama-server JSON response must be an object.",
                http_status=502,
                detail=type(data).__name__,
            )
            raise self._finalize_chat_error(active_proc, current_pid, err)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "llama-server response did not contain any completion choices.",
                http_status=502,
                detail=json.dumps(data, ensure_ascii=False)[:300],
            )
            raise self._finalize_chat_error(active_proc, current_pid, err)

        first_choice = choices[0]
        message_obj = first_choice.get("message")
        content: Any = None
        if isinstance(message_obj, dict):
            content = message_obj.get("content")
            reasoning = message_obj.get("reasoning_content")
            if (not isinstance(content, str) or not content.strip()) and isinstance(reasoning, str) and reasoning.strip():
                content = reasoning.strip()
        elif isinstance(first_choice.get("text"), str):
            content = first_choice.get("text")

        if not isinstance(content, str):
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "llama-server completion choice is missing text content.",
                http_status=502,
                detail=json.dumps(first_choice, ensure_ascii=False)[:300],
            )
            raise self._finalize_chat_error(active_proc, current_pid, err)

        usage = data.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        with self._state_lock:
            if self.mode == "managed_server":
                if (
                    active_proc is not None
                    and self._proc is active_proc
                    and self._status.state != "stopped"
                ):
                    alive_pid = active_proc.pid if active_proc.poll() is None else None
                    self._status = EngineStatus(
                        state="ready",
                        mode=self.mode,
                        endpoint=self.endpoint,
                        pid=alive_pid,
                        error_code="",
                        detail="",
                    )
            else:
                self._status = EngineStatus(
                    state="ready",
                    mode=self.mode,
                    endpoint=self.endpoint,
                    pid=None,
                    error_code="",
                    detail="",
                )

        return {
            "reply": content,
            "usage": usage,
            "model": str(data.get("model") or ""),
            "finish_reason": str(first_choice.get("finish_reason") or "stop"),
        }

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._terminate_proc_locked()
            self._set_status(state="stopped", pid=None, error_code="", detail="")


class LlamaCliEngine(LlamaEngine):
    """Fallback adapter invoking llama-cli per turn via subprocess."""

    def __init__(self, llama_config: Mapping[str, Any]) -> None:
        cfg = dict(llama_config)
        self.mode: str = "cli"
        self.llama_cpp_path: str = str(cfg.get("llama_cpp_path", ""))
        self.model_path: str = str(cfg.get("model_path", ""))
        self.mmproj_path: str = str(cfg.get("mmproj_path", "") or "")
        self.ctx_size: int = int(cfg.get("ctx_size", 16384))
        self.n_gpu_layers: int = int(cfg.get("n_gpu_layers", -1))
        self.request_timeout_sec: float = float(cfg.get("request_timeout_sec", 120))
        self.temperature: float = float(cfg.get("temperature", 0.8))
        self.top_p: float = float(cfg.get("top_p", 0.9))
        self.max_tokens: int = int(cfg.get("max_tokens", 512))

        self._lock = threading.Lock()
        self._resolved_exe: Path | None = None
        self._resolved_model: Path | None = None
        self._resolved_mmproj: Path | None = None
        self._status = EngineStatus(
            state="stopped",
            mode="cli",
            endpoint="",
            pid=None,
            error_code="",
            detail="",
        )

    def _set_status(
        self,
        *,
        state: EngineState,
        error_code: str = "",
        detail: str = "",
    ) -> None:
        with self._lock:
            self._status = EngineStatus(
                state=state,
                mode="cli",
                endpoint="",
                pid=None,
                error_code=error_code,
                detail=detail,
            )

    @staticmethod
    def format_messages(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_chunks = [
                    str(b.get("text", ""))
                    for b in content
                    if isinstance(b, Mapping) and b.get("type") == "text"
                ]
                rendered = "\n".join(c for c in text_chunks if c)
            else:
                rendered = str(content)
            parts.append(f"<|im_start|>{msg['role']}\n{rendered}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def status(self) -> EngineStatus:
        with self._lock:
            return dataclasses.replace(self._status)

    def ensure_ready(self) -> None:
        try:
            if not self.llama_cpp_path.strip() or not self.model_path.strip():
                raise EngineError(
                    "CONFIG_INCOMPLETE",
                    "Both llama_cpp_path and model_path are required in cli mode.",
                    http_status=400,
                )
            exe = resolve_executable(self.llama_cpp_path, _CLI_BIN_CANDIDATES)
            model = validate_gguf_model(self.model_path)
            mmproj = (
                validate_gguf_model(self.mmproj_path)
                if self.mmproj_path.strip()
                else None
            )
        except EngineError as exc:
            self._set_status(state="error", error_code=exc.code, detail=exc.detail or exc.message)
            raise

        with self._lock:
            self._resolved_exe = exe
            self._resolved_model = model
            self._resolved_mmproj = mmproj
            self._status = EngineStatus(
                state="ready",
                mode="cli",
                endpoint="",
                pid=None,
                error_code="",
                detail="",
            )

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_messages = _validate_messages(messages)
        temperature, top_p, max_tokens = _validate_sampling_params(
            params,
            default_temperature=self.temperature,
            default_top_p=self.top_p,
            default_max_tokens=self.max_tokens,
        )
        self.ensure_ready()

        with self._lock:
            exe = self._resolved_exe
            model = self._resolved_model
            mmproj = self._resolved_mmproj
        assert exe is not None and model is not None

        prompt = self.format_messages(normalized_messages)

        tmp_file: Path | None = None
        proc: subprocess.Popen[str] | None = None
        use_prompt_file = len(prompt) > _CLI_INLINE_PROMPT_LIMIT or (
            os.name == "nt"
            and exe.suffix.lower() in (".bat", ".cmd")
            and ("\n" in prompt or "\r" in prompt)
        )
        try:
            if not use_prompt_file:
                prompt_args = ["-p", prompt]
            else:
                fd, tmp_name = tempfile.mkstemp(prefix="moris_cli_prompt_", suffix=".txt")
                os.close(fd)
                tmp_file = Path(tmp_name)
                tmp_file.write_text(prompt, encoding="utf-8")
                prompt_args = ["-f", str(tmp_file)]

            cmd = [
                str(exe),
                "-m",
                str(model),
                *prompt_args,
                "-c",
                str(self.ctx_size),
                "-ngl",
                str(self.n_gpu_layers),
                "-n",
                str(max_tokens),
                "--temp",
                str(temperature),
                "--top-p",
                str(top_p),
                "--no-display-prompt",
            ]
            if mmproj is not None:
                cmd.extend(["--mmproj", str(mmproj)])
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
            )
            stdout_text, stderr_out = proc.communicate(
                timeout=max(0.05, float(self.request_timeout_sec))
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            capture_output=True,
                            timeout=5,
                            creationflags=_CREATE_NO_WINDOW,
                            check=False,
                        )
                    except Exception:
                        pass
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.communicate(timeout=2.0)
                except Exception:
                    pass
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None and not stream.closed:
                        try:
                            stream.close()
                        except Exception:
                            pass
            err = EngineError(
                "ENGINE_TIMEOUT",
                "llama-cli timed out while generating response.",
                http_status=504,
                detail=str(exc),
            )
            self._set_status(state="error", error_code=err.code, detail=err.detail)
            raise err from exc
        except (FileNotFoundError, PermissionError, OSError) as exc:
            if proc is not None:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None and not stream.closed:
                        try:
                            stream.close()
                        except Exception:
                            pass
            err = EngineError(
                "ENGINE_BINARY_NOT_FOUND",
                "Failed to execute llama-cli binary.",
                http_status=400,
                detail=f"{exe}: {exc}",
            )
            self._set_status(state="error", error_code=err.code, detail=err.detail)
            raise err from exc
        finally:
            if tmp_file is not None:
                try:
                    tmp_file.unlink(missing_ok=True)
                except OSError:
                    pass

        if returncode != 0:
            stderr_text = (stderr_out or "").strip()
            err = EngineError(
                "ENGINE_PROCESS_EXITED",
                f"llama-cli exited with code {returncode}.",
                http_status=502,
                detail=stderr_text,
            )
            self._set_status(state="error", error_code=err.code, detail=err.detail)
            raise err

        reply = stdout_text or ""
        if reply.startswith(prompt):
            reply = reply[len(prompt):]
        for marker in ("<|im_end|>", "[end of text]"):
            if marker in reply:
                reply = reply.split(marker, 1)[0]
        reply = reply.strip()

        if not reply:
            err = EngineError(
                "ENGINE_UPSTREAM_ERROR",
                "llama-cli returned empty output.",
                http_status=502,
                detail=(stderr_out or "").strip(),
            )
            self._set_status(state="error", error_code=err.code, detail=err.detail)
            raise err

        self._set_status(state="ready", error_code="", detail="")
        return {
            "reply": reply,
            "usage": {},
            "model": model.name,
            "finish_reason": "stop",
        }

    def shutdown(self) -> None:
        self._set_status(state="stopped", error_code="", detail="")


def create_engine(llama_config: Mapping[str, Any]) -> LlamaEngine:
    """Create a LlamaEngine instance from a llama sub-dict or full config dict."""
    if not isinstance(llama_config, Mapping):
        raise EngineError(
            "CONFIG_INCOMPLETE",
            "llama_config must be a mapping.",
            http_status=400,
        )

    if "engine_mode" not in llama_config and isinstance(llama_config.get("llama"), Mapping):
        cfg = llama_config["llama"]
    else:
        cfg = llama_config

    mode = str(cfg.get("engine_mode", "managed_server")).strip() or "managed_server"
    if mode in ("managed_server", "external_server"):
        return LlamaServerEngine(cfg)
    if mode == "cli":
        return LlamaCliEngine(cfg)
    raise EngineError(
        "CONFIG_INCOMPLETE",
        f"Unknown llama engine_mode: {mode!r}",
        http_status=400,
        detail=mode,
    )
