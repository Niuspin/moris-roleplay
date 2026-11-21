"""Pure-stdlib ThreadingHTTPServer REST API and static asset server for Moris Roleplay."""

from __future__ import annotations

import http.server
import json
import mimetypes
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from backend.chat_service import ChatError, ChatService
from backend.config_store import ConfigError, ConfigStore
from backend.extensions.t2i import T2IHookDispatcher, T2IRegistry
from backend.library_manager import LibraryManager
from backend.llama_engine import (
    EngineError,
    EngineStatus,
    create_engine,
    probe_tcp_port_available,
    reclaim_orphan_project_engine,
)
from backend.skill_loader import SkillLoader, SkillLoadError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIC_DIR = REPO_ROOT / "static"

_CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


def _status_to_dict(status_obj: Any) -> dict[str, Any]:
    if isinstance(status_obj, dict):
        return dict(status_obj)
    if isinstance(status_obj, EngineStatus) or hasattr(status_obj, "to_dict"):
        return status_obj.to_dict()
    return {"state": str(status_obj)}


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _resolve_existing_or_discover(raw_path: str, candidates: list[dict[str, Any]], key: str = "path") -> str:
    """Return raw_path if it resolves to an existing file; otherwise auto-match by filename or first candidate."""
    cleaned = (raw_path or "").strip()
    if cleaned:
        p = Path(cleaned).expanduser()
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            return raw_path
        target_name = p.name.lower()
        for item in candidates:
            cand_path = str(item.get(key, "") or "")
            if cand_path and Path(cand_path).name.lower() == target_name:
                return cand_path
    if candidates:
        return str(candidates[0].get(key, "") or "")
    return ""


def _to_portable_path(raw_path: str) -> str:
    """Convert paths inside REPO_ROOT to relative paths for portability."""
    cleaned = (raw_path or "").strip()
    if not cleaned:
        return ""
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        return cleaned
    try:
        return str(p.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return cleaned


def _connect_host_for(host: str) -> str:
    cleaned = (host or "").strip()
    if cleaned in ("", "0.0.0.0", "localhost"):
        return "127.0.0.1"
    if cleaned == "::":
        return "::1"
    return cleaned


def _probe_existing_llama_server(host: str, port: int, expected_model_path: str) -> tuple[bool, str]:
    """Check if host:port hosts a healthy llama-server serving the expected (or any) model."""
    c_host = _connect_host_for(host)
    url_host = f"[{c_host}]" if ":" in c_host and not c_host.startswith("[") else c_host
    base = f"http://{url_host}:{port}"

    try:
        req = urllib.request.Request(f"{base}/health", method="GET")
        with _NO_PROXY_OPENER.open(req, timeout=1.2) as resp:
            if getattr(resp, "status", 200) != 200:
                return False, base
            health_data = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
            if isinstance(health_data, dict):
                st = str(health_data.get("status", "")).lower()
                if st and st != "ok":
                    return False, base
    except Exception:
        return False, base

    # Verify it speaks llama-server /v1/models and check loaded model match
    try:
        m_req = urllib.request.Request(f"{base}/v1/models", method="GET")
        with _NO_PROXY_OPENER.open(m_req, timeout=1.2) as m_resp:
            m_data = json.loads(m_resp.read().decode("utf-8", errors="replace") or "{}")
    except Exception:
        return False, base

    models_list = m_data.get("models") if isinstance(m_data, dict) else None
    if not isinstance(models_list, list):
        models_list = m_data.get("data") if isinstance(m_data, dict) else None
    if not isinstance(models_list, list) or not models_list:
        return False, base

    if not expected_model_path.strip():
        return True, base

    exp_name = Path(expected_model_path.strip()).name.lower()
    for item in models_list:
        if not isinstance(item, dict):
            continue
        for key in ("model", "name", "id"):
            val = str(item.get(key, "") or "").strip()
            if val and (Path(val).name.lower() == exp_name or exp_name in val.lower()):
                return True, base
    return False, base


def _find_free_llama_port(host: str, preferred_port: int) -> int:
    """Find an available TCP port starting after preferred_port."""
    for candidate in range(preferred_port + 1, min(65536, preferred_port + 101)):
        try:
            probe_tcp_port_available(host, candidate)
            return candidate
        except EngineError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class MorisHTTPServer(http.server.ThreadingHTTPServer):
    """Multi-threaded HTTP server holding application services and static directory."""

    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[http.server.BaseHTTPRequestHandler],
        *,
        chat_service: ChatService,
        config_store: ConfigStore,
        t2i_dispatcher: T2IHookDispatcher,
        library_manager: LibraryManager | None = None,
        static_dir: Path = DEFAULT_STATIC_DIR,
    ) -> None:
        self.chat_service = chat_service
        self.config_store = config_store
        self.t2i_dispatcher = t2i_dispatcher
        self.library_manager = library_manager if library_manager is not None else LibraryManager()
        self.static_dir = Path(static_dir).resolve()
        super().__init__(server_address, request_handler_class)

    def server_bind(self) -> None:
        if os.name == "nt":
            exclusive_opt = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive_opt is not None:
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, exclusive_opt, 1)
                except OSError:
                    pass
        super().server_bind()


class MorisRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler serving static UI files and /api/* JSON endpoints."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr per-request noise to keep console output clean
        return

    @property
    def app_server(self) -> MorisHTTPServer:
        return self.server  # type: ignore[return-value]

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _send_error_json(
        self,
        status: int,
        code: str,
        message: str,
        detail: str = "",
        field: str = "",
    ) -> None:
        err_obj: dict[str, str] = {
            "code": code,
            "message": message,
            "detail": detail,
        }
        if field:
            err_obj["field"] = field
        self._send_json(status, {"error": err_obj})

    def _send_domain_exception(
        self,
        exc: ConfigError | SkillLoadError | EngineError | ChatError,
    ) -> None:
        status = int(getattr(exc, "http_status", 400))
        err_dict = exc.to_dict()
        err_dict.setdefault("detail", "")
        self._send_json(status, {"error": err_dict})

    def _stream_sse(self, event_iter: Any, cancel_event: threading.Event) -> None:
        """Stream an iterator of dicts as Server-Sent Events (text/event-stream)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def _write_frame(payload: dict[str, Any]) -> bool:
            chunk = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                cancel_event.set()
                return False

        try:
            for ev in event_iter:
                if not _write_frame(ev):
                    break
        except (ConfigError, SkillLoadError, EngineError, ChatError) as exc:
            err_dict = exc.to_dict()
            err_dict.setdefault("detail", "")
            _write_frame({"type": "error", "error": err_dict})
        except Exception as exc:
            _write_frame(
                {
                    "type": "error",
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "Unexpected streaming error.",
                        "detail": f"{type(exc).__name__}: {exc}",
                    },
                }
            )
        finally:
            cancel_event.set()

    def _read_json_body(self, *, allow_empty: bool = False) -> dict[str, Any]:
        raw_cl = self.headers.get("Content-Length")
        if raw_cl is None:
            if allow_empty:
                return {}
            raise ChatError(
                "INVALID_JSON_BODY",
                "Request body is required.",
                http_status=400,
            )

        try:
            content_length = int(raw_cl.strip())
        except ValueError as exc:
            raise ChatError(
                "INVALID_CONTENT_LENGTH",
                "Invalid Content-Length header.",
                detail=raw_cl,
                http_status=400,
            ) from exc

        if content_length < 0:
            raise ChatError(
                "INVALID_CONTENT_LENGTH",
                "Content-Length cannot be negative.",
                http_status=400,
            )

        if content_length == 0:
            if allow_empty:
                return {}
            raise ChatError(
                "EMPTY_REQUEST_BODY",
                "Request JSON body cannot be empty.",
                http_status=400,
            )

        raw_bytes = self.rfile.read(content_length)
        if not raw_bytes or not raw_bytes.strip():
            if allow_empty:
                return {}
            raise ChatError(
                "EMPTY_REQUEST_BODY",
                "Request JSON body cannot be empty.",
                http_status=400,
            )

        try:
            text = raw_bytes.decode("utf-8")
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChatError(
                "INVALID_JSON",
                "Malformed JSON in request body.",
                detail=str(exc),
                http_status=400,
            ) from exc

        if not isinstance(parsed, dict):
            raise ChatError(
                "INVALID_JSON_OBJECT",
                "Request JSON body must be a JSON object.",
                detail=f"Got {type(parsed).__name__}",
                http_status=400,
            )
        return parsed

    def _serve_static_file(self, rel_path: str) -> None:
        static_root = self.app_server.static_dir
        unquoted = urllib.parse.unquote(rel_path)

        # Reject explicit parent traversal segments or null bytes immediately
        normalized_slashes = unquoted.replace("\\", "/")
        segments = [seg for seg in normalized_slashes.split("/") if seg]
        if "\x00" in unquoted or ".." in segments:
            self._send_error_json(
                403,
                "FORBIDDEN_PATH",
                "Path traversal outside static directory is forbidden.",
            )
            return

        candidate = (static_root / "/".join(segments)).resolve()
        try:
            candidate.relative_to(static_root)
        except ValueError:
            self._send_error_json(
                403,
                "FORBIDDEN_PATH",
                "Path traversal outside static directory is forbidden.",
            )
            return

        if not candidate.is_file():
            self._send_error_json(
                404,
                "STATIC_ASSET_NOT_FOUND",
                f"Static file not found: {rel_path}",
            )
            return

        try:
            content = candidate.read_bytes()
        except OSError as exc:
            self._send_error_json(
                500,
                "STATIC_READ_ERROR",
                "Failed to read static asset.",
                detail=str(exc),
            )
            return

        ext = candidate.suffix.lower()
        content_type = _CONTENT_TYPES.get(ext)
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(candidate))
            content_type = guessed or "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(content)

    # ------------------------------------------------------------------
    # HTTP method dispatchers
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlsplit(self.path)
        path_only = parsed_url.path

        try:
            if path_only in ("/", "/index.html"):
                self._serve_static_file("index.html")
                return

            if path_only.startswith("/static/"):
                rel = path_only[len("/static/") :]
                self._serve_static_file(rel)
                return

            if path_only == "/api/health":
                cfg = self.app_server.chat_service.get_config()
                eng_status = _status_to_dict(self.app_server.chat_service.engine_status())
                skill_status = self.app_server.chat_service.current_skill_dict()
                t2i_status = self.app_server.t2i_dispatcher.status_summary(cfg.get("t2i"))
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "engine": eng_status,
                        "skill": skill_status,
                        "t2i": t2i_status,
                    },
                )
                return

            if path_only == "/api/config":
                cfg = self.app_server.chat_service.get_config()
                self._send_json(200, cfg)
                return

            if path_only == "/api/skill/current":
                self._send_json(200, self.app_server.chat_service.current_skill_dict())
                return

            if path_only == "/api/engine/status":
                st = _status_to_dict(self.app_server.chat_service.engine_status())
                self._send_json(200, st)
                return

            if path_only == "/api/extensions/t2i":
                cfg = self.app_server.chat_service.get_config()
                summary = self.app_server.t2i_dispatcher.status_summary(cfg.get("t2i"))
                self._send_json(200, summary)
                return

            if path_only == "/api/library":
                cfg = self.app_server.chat_service.get_config()
                overview = self.app_server.library_manager.scan_all(cfg)
                self._send_json(200, overview)
                return

            if path_only == "/api/sessions":
                sessions = self.app_server.chat_service.list_sessions()
                self._send_json(200, {"sessions": sessions})
                return

            if path_only == "/api/sessions/history":
                qs = urllib.parse.parse_qs(parsed_url.query)
                sid = (qs.get("session_id") or [""])[0]
                detail = self.app_server.chat_service.get_session_detail(sid)
                self._send_json(200, detail)
                return

            if path_only.startswith("/api/"):
                self._send_error_json(
                    404,
                    "API_NOT_FOUND",
                    f"Unknown API endpoint: {path_only}",
                    detail=path_only,
                )
                return

            self._send_error_json(
                404,
                "NOT_FOUND",
                f"Route not found: {path_only}",
                detail=path_only,
            )
        except (ConfigError, SkillLoadError, EngineError, ChatError) as exc:
            self._send_domain_exception(exc)
        except Exception as exc:
            self._send_error_json(
                500,
                "INTERNAL_SERVER_ERROR",
                "Unexpected internal server error.",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def do_PUT(self) -> None:
        parsed_url = urllib.parse.urlsplit(self.path)
        path_only = parsed_url.path

        try:
            if path_only == "/api/config":
                patch = self._read_json_body(allow_empty=False)
                updated = self.app_server.chat_service.update_config(patch)
                self.app_server.t2i_dispatcher.refresh_plugins(updated.get("t2i"))
                self._send_json(200, updated)
                return

            if path_only == "/api/sessions/message":
                body = self._read_json_body(allow_empty=False)
                res = self.app_server.chat_service.edit_message(
                    session_id=body.get("session_id", ""),
                    index=body.get("index", -1),
                    content=body.get("content", ""),
                )
                self._send_json(200, res)
                return

            self._send_error_json(
                404,
                "API_NOT_FOUND",
                f"Unknown PUT endpoint: {path_only}",
                detail=path_only,
            )
        except (ConfigError, SkillLoadError, EngineError, ChatError) as exc:
            self._send_domain_exception(exc)
        except Exception as exc:
            self._send_error_json(
                500,
                "INTERNAL_SERVER_ERROR",
                "Unexpected internal server error.",
                detail=f"{type(exc).__name__}: {exc}",
            )

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlsplit(self.path)
        path_only = parsed_url.path

        try:
            if path_only == "/api/skill/load":
                body = self._read_json_body(allow_empty=False)
                raw_path = body.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise SkillLoadError(
                        "SKILL_PATH_REQUIRED",
                        "Field 'path' must be a non-empty string.",
                        http_status=400,
                    )
                exclude_meta = body.get("exclude_meta_docs")
                max_chars = body.get("max_prompt_chars")
                loaded = self.app_server.chat_service.load_skill(
                    path=raw_path,
                    exclude_meta_docs=exclude_meta,
                    max_prompt_chars=max_chars,
                )
                self._send_json(200, {"loaded": True, **loaded.to_dict()})
                return

            if path_only == "/api/engine/start":
                self._read_json_body(allow_empty=True)
                st = _status_to_dict(self.app_server.chat_service.start_engine())
                self._send_json(200, st)
                return

            if path_only == "/api/engine/stop":
                self._read_json_body(allow_empty=True)
                st = _status_to_dict(self.app_server.chat_service.stop_engine())
                self._send_json(200, st)
                return

            if path_only == "/api/chat":
                body = self._read_json_body(allow_empty=False)
                msg = body.get("message")
                session_id = body.get("session_id")
                params = body.get("params")
                images = body.get("images")
                chat_res = self.app_server.chat_service.chat(
                    message=msg,  # type: ignore[arg-type]
                    session_id=session_id,
                    params=params,
                    images=images,
                )
                self._send_json(200, chat_res)
                return

            if path_only == "/api/chat/stream":
                body = self._read_json_body(allow_empty=False)
                msg = body.get("message")
                session_id = body.get("session_id")
                params = body.get("params")
                images = body.get("images")
                cancel_event = threading.Event()
                event_iter = self.app_server.chat_service.chat_stream(
                    message=msg,  # type: ignore[arg-type]
                    session_id=session_id,
                    params=params,
                    images=images,
                    cancel_event=cancel_event,
                )
                self._stream_sse(event_iter, cancel_event)
                return

            if path_only == "/api/chat/regenerate/stream":
                body = self._read_json_body(allow_empty=False)
                session_id = body.get("session_id", "")
                params = body.get("params")
                cancel_event = threading.Event()
                event_iter = self.app_server.chat_service.regenerate_stream(
                    session_id=session_id,
                    params=params,
                    cancel_event=cancel_event,
                )
                self._stream_sse(event_iter, cancel_event)
                return

            if path_only == "/api/sessions/truncate":
                body = self._read_json_body(allow_empty=False)
                res = self.app_server.chat_service.delete_messages_from(
                    session_id=body.get("session_id", ""),
                    index=body.get("index", -1),
                )
                self._send_json(200, res)
                return

            if path_only == "/api/sessions/delete":
                body = self._read_json_body(allow_empty=False)
                res = self.app_server.chat_service.delete_session(
                    session_id=body.get("session_id", ""),
                )
                self._send_json(200, res)
                return

            if path_only == "/api/quick-ready":
                self._read_json_body(allow_empty=True)
                cfg = self.app_server.chat_service.get_config()
                overview = self.app_server.library_manager.scan_all(cfg)
                llama_cfg = cfg.get("llama", {})

                # Ensure local engine & model defaults are populated / auto-repaired from local libraries
                eng_items = overview.get("engines", {}).get("items", [])
                model_items = overview.get("models", {}).get("models", [])
                mmproj_items = overview.get("models", {}).get("mmprojs", [])

                fill_patch: dict[str, Any] = {}
                cur_exe = str(llama_cfg.get("llama_cpp_path", "") or "")
                resolved_exe = _resolve_existing_or_discover(cur_exe, eng_items)
                if resolved_exe and resolved_exe != cur_exe:
                    fill_patch["llama_cpp_path"] = _to_portable_path(resolved_exe)

                cur_model = str(llama_cfg.get("model_path", "") or "")
                resolved_model = _resolve_existing_or_discover(cur_model, model_items)
                if resolved_model and resolved_model != cur_model:
                    fill_patch["model_path"] = _to_portable_path(resolved_model)
                    matched_entry = next(
                        (m for m in model_items if m.get("path") == resolved_model), None
                    )
                    if matched_entry and matched_entry.get("suggested_mmproj"):
                        fill_patch["mmproj_path"] = _to_portable_path(
                            str(matched_entry["suggested_mmproj"])
                        )
                elif cur_model:
                    cur_mm = str(llama_cfg.get("mmproj_path", "") or "")
                    if cur_mm:
                        resolved_mm = _resolve_existing_or_discover(cur_mm, mmproj_items)
                        if resolved_mm and resolved_mm != cur_mm:
                            fill_patch["mmproj_path"] = _to_portable_path(resolved_mm)

                if fill_patch:
                    cfg = self.app_server.chat_service.update_config({"llama": fill_patch})
                    llama_cfg = cfg.get("llama", {})

                # Restore last skill or pick first available skill if none loaded
                if self.app_server.chat_service.current_skill() is None:
                    restored = self.app_server.chat_service.restore_last_skill()
                    if restored is None:
                        skill_items = overview.get("skills", {}).get("items", [])
                        if skill_items:
                            try:
                                self.app_server.chat_service.load_skill(skill_items[0]["path"])
                            except Exception:
                                pass

                notice = ""
                mode = str(llama_cfg.get("engine_mode", "managed_server")).strip() or "managed_server"
                cur_status = _status_to_dict(self.app_server.chat_service.engine_status())

                if cur_status.get("state") == "ready":
                    eng_status = cur_status
                elif mode == "external_server":
                    try:
                        eng_status = _status_to_dict(self.app_server.chat_service.start_engine())
                    except EngineError:
                        # External server unavailable; fall back to managed_server if local engine + model exist
                        if str(llama_cfg.get("llama_cpp_path", "")).strip() and str(llama_cfg.get("model_path", "")).strip():
                            l_host = str(llama_cfg.get("host", "127.0.0.1"))
                            l_port = int(llama_cfg.get("port", 8080))
                            try:
                                probe_tcp_port_available(l_host, l_port)
                                target_port = l_port
                            except EngineError:
                                reclaimed = reclaim_orphan_project_engine(l_port, llama_cfg.get("engine_dir"))
                                if reclaimed is not None:
                                    target_port = l_port
                                else:
                                    target_port = _find_free_llama_port(l_host, l_port)
                            self.app_server.chat_service.update_config(
                                {"llama": {"engine_mode": "managed_server", "port": target_port}}
                            )
                            eng_status = _status_to_dict(self.app_server.chat_service.start_engine())
                            notice = f"已自动切换为本程序托管引擎并启动 (端口 {target_port})"
                        else:
                            raise
                elif mode == "managed_server":
                    l_host = str(llama_cfg.get("host", "127.0.0.1"))
                    l_port = int(llama_cfg.get("port", 8080))
                    try:
                        probe_tcp_port_available(l_host, l_port)
                    except EngineError as port_exc:
                        if port_exc.code != "ENGINE_PORT_CONFLICT":
                            raise
                        reclaimed_pid = reclaim_orphan_project_engine(
                            l_port, llama_cfg.get("engine_dir")
                        )
                        if reclaimed_pid is not None:
                            notice = f"已自动清理残留引擎进程 (PID {reclaimed_pid}) 并重新拉起本程序引擎 (端口 {l_port})"
                        else:
                            free_port = _find_free_llama_port(l_host, l_port)
                            self.app_server.chat_service.update_config(
                                {"llama": {"port": free_port}}
                            )
                            notice = f"端口 {l_port} 已被其他程序占用，已自动切换至空闲端口 {free_port} 拉起本程序引擎"
                    eng_status = _status_to_dict(self.app_server.chat_service.start_engine())
                else:
                    eng_status = _status_to_dict(self.app_server.chat_service.start_engine())

                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "notice": notice,
                        "engine": eng_status,
                        "skill": self.app_server.chat_service.current_skill_dict(),
                        "config": self.app_server.chat_service.get_config(),
                    },
                )
                return

            if path_only == "/api/library/rescan":
                body = self._read_json_body(allow_empty=True)
                patch: dict[str, Any] = {}
                if "skill_library_dirs" in body:
                    patch.setdefault("skill", {})["library_dirs"] = body["skill_library_dirs"]
                if "engine_dir" in body:
                    patch.setdefault("llama", {})["engine_dir"] = body["engine_dir"]
                if "models_dir" in body:
                    patch.setdefault("llama", {})["models_dir"] = body["models_dir"]
                if patch:
                    cfg = self.app_server.chat_service.update_config(patch)
                else:
                    cfg = self.app_server.chat_service.get_config()
                overview = self.app_server.library_manager.scan_all(cfg)
                self._send_json(200, overview)
                return

            if path_only == "/api/sessions/reset":
                body = self._read_json_body(allow_empty=True)
                session_id = body.get("session_id")
                reset_res = self.app_server.chat_service.reset_session(session_id=session_id)
                self._send_json(200, reset_res)
                return

            self._send_error_json(
                404,
                "API_NOT_FOUND",
                f"Unknown POST endpoint: {path_only}",
                detail=path_only,
            )
        except (ConfigError, SkillLoadError, EngineError, ChatError) as exc:
            self._send_domain_exception(exc)
        except Exception as exc:
            self._send_error_json(
                500,
                "INTERNAL_SERVER_ERROR",
                "Unexpected internal server error.",
                detail=f"{type(exc).__name__}: {exc}",
            )


    def do_DELETE(self) -> None:
        parsed_url = urllib.parse.urlsplit(self.path)
        path_only = parsed_url.path
        try:
            if path_only == "/api/sessions":
                qs = urllib.parse.parse_qs(parsed_url.query)
                sid = (qs.get("session_id") or [""])[0]
                res = self.app_server.chat_service.delete_session(sid)
                self._send_json(200, res)
                return

            self._send_error_json(
                404,
                "API_NOT_FOUND",
                f"Unknown DELETE endpoint: {path_only}",
                detail=path_only,
            )
        except (ConfigError, SkillLoadError, EngineError, ChatError) as exc:
            self._send_domain_exception(exc)
        except Exception as exc:
            self._send_error_json(
                500,
                "INTERNAL_SERVER_ERROR",
                "Unexpected internal server error.",
                detail=f"{type(exc).__name__}: {exc}",
            )


def build_application_server(
    *,
    config_path: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
    static_dir: Path = DEFAULT_STATIC_DIR,
    restore_skill: bool = False,
) -> MorisHTTPServer:
    """Wire ConfigStore, T2IRegistry/Dispatcher, ChatService, and MorisHTTPServer."""
    resolved_config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else (REPO_ROOT / "config.json").resolve()
    )
    example_path = REPO_ROOT / "config.example.json"
    config_store = ConfigStore(
        resolved_config_path,
        example_path=example_path if example_path.is_file() else None,
    )
    cfg = config_store.load()

    bind_host = (host.strip() if isinstance(host, str) and host.strip() else str(cfg["server"]["host"]))
    bind_port = int(port if port is not None else cfg["server"]["port"])
    if not (1 <= bind_port <= 65535):
        raise ConfigError(
            "CONFIG_VALIDATION_ERROR",
            f"Server port must be between 1 and 65535, got {bind_port}.",
            field="server.port",
            http_status=400,
        )

    # Probe TCP port availability before binding to fail fast on conflicts
    probe_tcp_port_available(bind_host, bind_port)

    registry = T2IRegistry(register_builtins=True)
    dispatcher = T2IHookDispatcher(registry=registry, config_getter=config_store.load)
    dispatcher.refresh_plugins(cfg.get("t2i"))

    sessions_dir = resolved_config_path.parent / "sessions"
    chat_service = ChatService(
        engine_factory_or_engine=create_engine,
        config_store=config_store,
        skill_loader=SkillLoader,
        t2i_dispatcher=dispatcher,
        sessions_dir=sessions_dir,
    )

    if restore_skill:
        chat_service.restore_last_skill()

    library_manager = LibraryManager(repo_root=REPO_ROOT)
    library_manager.ensure_default_directories()

    return MorisHTTPServer(
        (bind_host, bind_port),
        MorisRequestHandler,
        chat_service=chat_service,
        config_store=config_store,
        t2i_dispatcher=dispatcher,
        library_manager=library_manager,
        static_dir=static_dir,
    )
