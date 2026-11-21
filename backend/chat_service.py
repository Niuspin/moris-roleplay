"""Stateful multi-turn roleplay chat service with skill injection and T2I hook dispatch."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.extensions.t2i import SceneContext
from backend.llama_engine import EngineStatus
from backend.skill_loader import SkillLoader, SkillLoadError, SkillLoadResult

_SAFE_SID_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_IMMERSION_DIRECTIVE = (
    "【沉浸式角色扮演指引】\n"
    "1. 全程保持角色设定中的性格、语气、口癖、知识边界与世界观，绝不跳戏或提及自己是 AI、语言模型或助手。\n"
    "2. 自然融入动作、神态、眼神、停顿与环境氛围描写（可使用 *动作或心理描写* 与对话结合），让场景具有画面感与情绪张力。\n"
    "3. 根据对方的言行产生真实的情感递进与记忆连贯，避免机械复读上一轮句式。"
)

_ENGINE_BINDING_KEYS: tuple[str, ...] = (
    "engine_mode",
    "llama_cpp_path",
    "model_path",
    "mmproj_path",
    "server_url",
    "host",
    "port",
    "ctx_size",
    "n_gpu_layers",
    "startup_timeout_sec",
    "request_timeout_sec",
)


class ChatError(Exception):
    """Structured error raised by ChatService for non-crashing API responses."""

    def __init__(
        self,
        code: str,
        message: str,
        detail: str = "",
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.http_status = http_status

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }

    def to_error_payload(self) -> dict[str, dict[str, str]]:
        return {"error": self.to_dict()}


@dataclass
class ChatSession:
    """Per-session conversation state with isolated turn lock."""

    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    turn_index: int = 0
    title: str = ""
    character_name: str = ""
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    lock: threading.RLock = field(default_factory=threading.RLock)


def _disabled_t2i_hook(t2i_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "disabled",
        "plugin_name": str(t2i_cfg.get("active_plugin", "noop")),
        "invoked": False,
        "prompt_used": "",
        "image_url": None,
        "image_path": None,
        "elapsed_ms": 0.0,
        "message": "T2I dispatcher not configured.",
        "extra": {},
    }


def _normalize_t2i_result(
    raw_result: Any,
    *,
    default_plugin: str,
    elapsed_ms: float,
) -> dict[str, Any]:
    if raw_result is None:
        data: dict[str, Any] = {}
    elif isinstance(raw_result, dict):
        data = dict(raw_result)
    elif hasattr(raw_result, "to_dict") and callable(raw_result.to_dict):
        converted = raw_result.to_dict()
        data = dict(converted) if isinstance(converted, dict) else {}
    elif is_dataclass(raw_result) and not isinstance(raw_result, type):
        data = asdict(raw_result)
    else:
        data = {
            "status": getattr(raw_result, "status", "stub_ok"),
            "plugin_name": getattr(raw_result, "plugin_name", default_plugin),
            "invoked": getattr(raw_result, "invoked", True),
            "prompt_used": getattr(raw_result, "prompt_used", ""),
            "image_url": getattr(raw_result, "image_url", None),
            "image_path": getattr(raw_result, "image_path", None),
            "elapsed_ms": getattr(raw_result, "elapsed_ms", elapsed_ms),
            "message": getattr(raw_result, "message", ""),
            "extra": getattr(raw_result, "extra", {}),
        }

    extra_val = data.get("extra")
    raw_elapsed = data.get("elapsed_ms")
    resolved_elapsed = elapsed_ms if raw_elapsed is None else float(raw_elapsed)
    return {
        "status": str(data.get("status", "stub_ok")),
        "plugin_name": str(data.get("plugin_name", default_plugin)),
        "invoked": bool(data.get("invoked", False)),
        "prompt_used": str(data.get("prompt_used", "") or ""),
        "image_url": data.get("image_url"),
        "image_path": data.get("image_path"),
        "elapsed_ms": resolved_elapsed,
        "message": str(data.get("message", "") or ""),
        "extra": dict(extra_val) if isinstance(extra_val, dict) else {},
    }


def _extract_reply_and_usage(raw_res: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw_res, dict):
        raise ChatError(
            "ENGINE_UPSTREAM_ERROR",
            "Engine returned a non-dict completion response.",
            detail=type(raw_res).__name__,
            http_status=502,
        )

    reply: Any = raw_res.get("reply")
    if not isinstance(reply, str):
        choices = raw_res.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                reply = msg.get("content")
            elif isinstance(choices[0].get("text"), str):
                reply = choices[0].get("text")

    if not isinstance(reply, str):
        raise ChatError(
            "ENGINE_UPSTREAM_ERROR",
            "Engine returned an invalid or empty completion response.",
            http_status=502,
        )

    raw_usage = raw_res.get("usage")
    usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    return reply, usage


class ChatService:
    """Multi-turn roleplay chat orchestrator integrating config, skills, engine, and T2I hooks."""

    def __init__(
        self,
        engine_factory_or_engine: Any,
        config_store: Any,
        skill_loader: Any = SkillLoader,
        t2i_dispatcher: Any = None,
        sessions_dir: str | Path | None = None,
    ) -> None:
        if hasattr(engine_factory_or_engine, "chat_completion") and hasattr(
            engine_factory_or_engine, "status"
        ):
            self._engine: Any = engine_factory_or_engine
            self._engine_factory: Any = None
        elif callable(engine_factory_or_engine):
            self._engine = None
            self._engine_factory = engine_factory_or_engine
        else:
            raise TypeError(
                "engine_factory_or_engine must be a LlamaEngine instance or a callable factory."
            )

        self._config_store = config_store
        self._skill_loader = skill_loader
        self._t2i_dispatcher = t2i_dispatcher
        self._sessions_dir: Path | None = Path(sessions_dir) if sessions_dir is not None else None

        self._state_lock = threading.RLock()
        self._engine_lock = threading.RLock()
        self._sessions: dict[str, ChatSession] = {}
        self._current_skill: SkillLoadResult | None = None
        self._engine_fingerprint: tuple[tuple[str, Any], ...] | None = None
        if self._sessions_dir is not None:
            self._load_persisted_sessions()

    @staticmethod
    def _compute_engine_fingerprint(cfg: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        llama_cfg = cfg.get("llama", {})
        return tuple((key, llama_cfg.get(key)) for key in _ENGINE_BINDING_KEYS)

    @staticmethod
    def _safe_shutdown_engine(engine: Any) -> None:
        if engine is not None and hasattr(engine, "shutdown"):
            try:
                engine.shutdown()
            except Exception:
                pass

    def _invoke_engine_factory(self, cfg: dict[str, Any]) -> Any:
        assert self._engine_factory is not None
        sig = inspect.signature(self._engine_factory)
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        if not positional_params and not has_var_positional:
            return self._engine_factory()

        if positional_params and positional_params[0].name in ("config", "full_config"):
            return self._engine_factory(copy.deepcopy(cfg))
        return self._engine_factory(copy.deepcopy(cfg.get("llama", {})))

    def _get_or_refresh_engine(self, cfg: dict[str, Any]) -> Any:
        with self._engine_lock:
            if self._engine_factory is None:
                return self._engine

            fingerprint = self._compute_engine_fingerprint(cfg)
            if self._engine is None or self._engine_fingerprint != fingerprint:
                if self._engine is not None:
                    self._safe_shutdown_engine(self._engine)
                self._engine = self._invoke_engine_factory(cfg)
                self._engine_fingerprint = fingerprint
            return self._engine

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def load_skill(
        self,
        path: str | Path,
        exclude_meta_docs: bool | None = None,
        max_prompt_chars: int | None = None,
        *,
        persist_config: bool = True,
    ) -> SkillLoadResult:
        """Load a character skill file or directory and activate it across sessions."""
        cfg = self._config_store.load()
        skill_cfg = cfg.get("skill", {})
        eff_exclude: Any = (
            bool(skill_cfg.get("exclude_meta_docs", False))
            if exclude_meta_docs is None
            else exclude_meta_docs
        )
        eff_max_chars: Any = (
            int(skill_cfg.get("max_prompt_chars", 0))
            if max_prompt_chars is None
            else max_prompt_chars
        )

        result = self._skill_loader.load(
            path,
            exclude_meta_docs=eff_exclude,
            max_prompt_chars=eff_max_chars,
        )

        with self._state_lock:
            self._current_skill = result

        if persist_config:
            self._config_store.update({"skill": {"last_path": result.source_path}})

        return result

    def current_skill(self) -> SkillLoadResult | None:
        """Return the currently active SkillLoadResult, or None if no skill is loaded."""
        with self._state_lock:
            return self._current_skill

    def current_skill_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the currently loaded skill."""
        with self._state_lock:
            skill = self._current_skill
        if skill is None:
            return {"loaded": False}
        return {"loaded": True, **skill.to_dict()}

    def restore_last_skill(self) -> SkillLoadResult | None:
        """Best-effort load of config['skill']['last_path'] without crashing on stale paths."""
        cfg = self._config_store.load()
        last_path = str(cfg.get("skill", {}).get("last_path", "")).strip()
        if not last_path:
            return None
        try:
            return self.load_skill(last_path, persist_config=False)
        except (SkillLoadError, OSError):
            return None

    # ------------------------------------------------------------------
    # Engine & config lifecycle
    # ------------------------------------------------------------------

    def engine_status(self) -> EngineStatus | dict[str, Any]:
        """Return current engine status without triggering startup."""
        cfg = self._config_store.load()
        engine = self._get_or_refresh_engine(cfg)
        return engine.status()

    def start_engine(self) -> EngineStatus | dict[str, Any]:
        """Ensure the configured engine is started/connected and return its status."""
        cfg = self._config_store.load()
        engine = self._get_or_refresh_engine(cfg)
        engine.ensure_ready()
        return engine.status()

    def stop_engine(self) -> EngineStatus | dict[str, Any]:
        """Stop any managed engine process and return its status."""
        with self._engine_lock:
            engine = self._engine
        if engine is not None:
            engine.shutdown()
            return engine.status()

        cfg = self._config_store.load()
        engine = self._get_or_refresh_engine(cfg)
        engine.shutdown()
        return engine.status()

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration from ConfigStore."""
        return self._config_store.load()

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Atomically persist a config patch and refresh engine bindings if needed."""
        self._config_store.update(patch)
        return self.reload_config()

    def reload_config(self) -> dict[str, Any]:
        """Reload config from ConfigStore and swap factory engine if bindings changed."""
        cfg = self._config_store.load()
        self._get_or_refresh_engine(cfg)
        return cfg

    def shutdown(self) -> None:
        """Idempotently shut down the active engine instance."""
        with self._engine_lock:
            engine = self._engine
            self._engine = None
            self._engine_fingerprint = None
        self._safe_shutdown_engine(engine)

    # ------------------------------------------------------------------
    # Session persistence & management
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _session_file_path(self, session_id: str) -> Path | None:
        if self._sessions_dir is None:
            return None
        safe_id = _SAFE_SID_RE.sub("_", session_id).strip("._") or "session"
        return self._sessions_dir / f"{safe_id[:96]}.json"

    def _load_persisted_sessions(self) -> None:
        if self._sessions_dir is None or not self._sessions_dir.is_dir():
            return
        try:
            files = sorted(self._sessions_dir.glob("*.json"))
        except OSError:
            return
        for fpath in files:
            try:
                raw = json.loads(fpath.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                sid = str(raw.get("session_id") or "").strip()
                if not sid:
                    continue
                raw_msgs = raw.get("messages")
                msgs: list[dict[str, str]] = []
                if isinstance(raw_msgs, list):
                    for m in raw_msgs:
                        if isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str):
                            msgs.append({"role": m["role"], "content": m["content"]})
                sess = ChatSession(
                    session_id=sid,
                    messages=msgs,
                    turn_index=int(raw.get("turn_index") or (len(msgs) // 2)),
                    title=str(raw.get("title") or ""),
                    character_name=str(raw.get("character_name") or ""),
                    updated_at=str(raw.get("updated_at") or self._now_iso()),
                )
                with self._state_lock:
                    self._sessions[sid] = sess
            except Exception:
                continue

    def _persist_session(self, session: ChatSession) -> None:
        fpath = self._session_file_path(session.session_id)
        if fpath is None:
            return
        payload = {
            "session_id": session.session_id,
            "title": session.title,
            "character_name": session.character_name,
            "turn_index": session.turn_index,
            "updated_at": session.updated_at,
            "messages": [{"role": m["role"], "content": m["content"]} for m in session.messages],
        }
        tmp_path: str | None = None
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=fpath.parent,
                prefix=f".{fpath.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_f:
                tmp_path = tmp_f.name
                tmp_f.write(serialized)
                tmp_f.flush()
            os.replace(tmp_path, fpath)
            tmp_path = None
        except OSError:
            pass
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _delete_persisted_session_file(self, session_id: str) -> None:
        fpath = self._session_file_path(session_id)
        if fpath is not None:
            try:
                fpath.unlink(missing_ok=True)
            except OSError:
                pass

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        with self._state_lock:
            session = self._sessions.get(session_id)
            if session is None:
                char_name = self._current_skill.character_name if self._current_skill is not None else ""
                session = ChatSession(
                    session_id=session_id,
                    character_name=char_name,
                    updated_at=self._now_iso(),
                )
                self._sessions[session_id] = session
            return session

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all sessions sorted by most recently updated."""
        with self._state_lock:
            sessions = list(self._sessions.values())
        summaries: list[dict[str, Any]] = []
        for sess in sessions:
            with sess.lock:
                preview = ""
                if sess.messages:
                    preview = sess.messages[-1]["content"][:80].replace("\n", " ").strip()
                title = sess.title
                if not title and sess.messages:
                    first_user = next((m["content"] for m in sess.messages if m["role"] == "user"), "")
                    title = first_user[:28].replace("\n", " ").strip()
                summaries.append(
                    {
                        "session_id": sess.session_id,
                        "title": title or "新对话",
                        "character_name": sess.character_name,
                        "turn_index": sess.turn_index,
                        "messages_count": len(sess.messages),
                        "updated_at": sess.updated_at,
                        "preview": preview,
                    }
                )
        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return summaries

    def get_session_detail(self, session_id: str) -> dict[str, Any]:
        """Return full metadata and messages for a session."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a non-empty string.",
                http_status=400,
            )
        sid = session_id.strip()
        with self._state_lock:
            session = self._sessions.get(sid)
        if session is None:
            return {
                "session_id": sid,
                "title": "新对话",
                "character_name": "",
                "turn_index": 0,
                "messages_count": 0,
                "updated_at": self._now_iso(),
                "messages": [],
            }
        with session.lock:
            return {
                "session_id": session.session_id,
                "title": session.title or "新对话",
                "character_name": session.character_name,
                "turn_index": session.turn_index,
                "messages_count": len(session.messages),
                "updated_at": session.updated_at,
                "messages": [dict(m) for m in session.messages],
            }

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Permanently remove a session from memory and disk."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a non-empty string.",
                http_status=400,
            )
        sid = session_id.strip()
        with self._state_lock:
            existed = self._sessions.pop(sid, None) is not None
        self._delete_persisted_session_file(sid)
        return {"status": "ok", "session_id": sid, "deleted": existed}

    def edit_message(self, session_id: str, index: int, content: str) -> dict[str, Any]:
        """Edit the text content of a specific message in a session."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ChatError("CHAT_INVALID_SESSION_ID", "Field 'session_id' must be a non-empty string.", http_status=400)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ChatError("CHAT_INVALID_ARGUMENT", "Field 'index' must be a non-negative integer.", http_status=400)
        if not isinstance(content, str) or not content.strip():
            raise ChatError("CHAT_INVALID_MESSAGE", "Field 'content' must be a non-empty string.", http_status=400)
        sid = session_id.strip()
        with self._state_lock:
            session = self._sessions.get(sid)
        if session is None:
            raise ChatError("SESSION_NOT_FOUND", f"Session '{sid}' not found.", http_status=404)
        with session.lock:
            if index >= len(session.messages):
                raise ChatError("MESSAGE_INDEX_OUT_OF_RANGE", f"Message index {index} out of range.", http_status=400)
            session.messages[index]["content"] = content.strip()
            session.updated_at = self._now_iso()
            self._persist_session(session)
            return {
                "status": "ok",
                "session_id": sid,
                "index": index,
                "messages_count": len(session.messages),
                "messages": [dict(m) for m in session.messages],
            }

    def delete_messages_from(self, session_id: str, index: int) -> dict[str, Any]:
        """Truncate session history starting from message index (for roleplay rewind/branching)."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ChatError("CHAT_INVALID_SESSION_ID", "Field 'session_id' must be a non-empty string.", http_status=400)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ChatError("CHAT_INVALID_ARGUMENT", "Field 'index' must be a non-negative integer.", http_status=400)
        sid = session_id.strip()
        with self._state_lock:
            session = self._sessions.get(sid)
        if session is None:
            raise ChatError("SESSION_NOT_FOUND", f"Session '{sid}' not found.", http_status=404)
        with session.lock:
            if index >= len(session.messages):
                raise ChatError("MESSAGE_INDEX_OUT_OF_RANGE", f"Message index {index} out of range.", http_status=400)
            del session.messages[index:]
            session.turn_index = sum(1 for m in session.messages if m["role"] == "assistant")
            session.updated_at = self._now_iso()
            self._persist_session(session)
            return {
                "status": "ok",
                "session_id": sid,
                "turn_index": session.turn_index,
                "messages_count": len(session.messages),
                "messages": [dict(m) for m in session.messages],
            }

    def reset_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Clear conversation turns for one or all sessions while keeping the active skill."""
        if session_id is not None and not isinstance(session_id, str):
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a string.",
                http_status=400,
            )

        if session_id is not None and session_id.strip():
            sid = session_id.strip()
            with self._state_lock:
                session = self._sessions.get(sid)
                skill_loaded = self._current_skill is not None
            if session is not None:
                with session.lock:
                    session.messages.clear()
                    session.turn_index = 0
                    session.updated_at = self._now_iso()
                    self._persist_session(session)
            return {
                "status": "ok",
                "session_id": sid,
                "turn_index": 0,
                "messages_count": 0,
                "skill_loaded": skill_loaded,
            }

        with self._state_lock:
            sessions = list(self._sessions.values())
            skill_loaded = self._current_skill is not None

        for session in sessions:
            with session.lock:
                session.messages.clear()
                session.turn_index = 0
                session.updated_at = self._now_iso()
                self._persist_session(session)

        return {
            "status": "ok",
            "session_id": None,
            "cleared_sessions": len(sessions),
            "skill_loaded": skill_loaded,
        }

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """Return a shallow copy of stored user/assistant messages for a session."""
        if not isinstance(session_id, str):
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a string.",
                http_status=400,
            )
        sid = session_id.strip()
        if not sid:
            return []
        with self._state_lock:
            session = self._sessions.get(sid)
        if session is None:
            return []
        with session.lock:
            return [dict(m) for m in session.messages]

    def _dispatch_t2i_hook(
        self,
        *,
        session_id: str,
        turn_index: int,
        skill: SkillLoadResult | None,
        user_message: str,
        assistant_reply: str,
        session_messages: list[dict[str, str]],
        messages_count: int,
        t2i_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        if self._t2i_dispatcher is None:
            return _disabled_t2i_hook(t2i_cfg)

        ctx_turns = int(t2i_cfg.get("context_window_turns", 4))
        if ctx_turns > 0:
            recent_slice = session_messages[-(ctx_turns * 2) :]
            recent_messages: tuple[dict[str, str], ...] = tuple(
                {"role": m["role"], "content": m["content"]} for m in recent_slice
            )
        else:
            recent_messages = ()

        character_name = skill.character_name if skill is not None else ""
        skill_source = skill.source_path if skill is not None else ""
        snippet = assistant_reply[:200].strip()
        scene_prompt_hint = f"{character_name}: {snippet}" if character_name else snippet

        scene_ctx = SceneContext(
            session_id=session_id,
            turn_index=turn_index,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            character_name=character_name,
            skill_source=skill_source,
            user_message=user_message,
            assistant_reply=assistant_reply,
            recent_messages=recent_messages,
            scene_prompt_hint=scene_prompt_hint,
            metadata={
                "messages_count": messages_count,
                "skill_loaded": skill is not None,
                "source_type": skill.source_type if skill is not None else "",
            },
        )

        default_plugin = str(t2i_cfg.get("active_plugin", "noop"))
        started = time.perf_counter()
        try:
            dispatch_fn = self._t2i_dispatcher.dispatch
            sig = inspect.signature(dispatch_fn)
            params = sig.parameters
            positional_params = [
                p
                for p in params.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            t2i_param = params.get("t2i_config")

            if t2i_param is not None:
                if t2i_param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    raw_result = dispatch_fn(scene_ctx, copy.deepcopy(t2i_cfg))
                else:
                    raw_result = dispatch_fn(scene_ctx, t2i_config=copy.deepcopy(t2i_cfg))
            elif len(positional_params) >= 2:
                raw_result = dispatch_fn(scene_ctx, copy.deepcopy(t2i_cfg))
            elif has_var_keyword:
                raw_result = dispatch_fn(scene_ctx, t2i_config=copy.deepcopy(t2i_cfg))
            else:
                raw_result = dispatch_fn(scene_ctx)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return _normalize_t2i_result(
                raw_result,
                default_plugin=default_plugin,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return {
                "status": "error",
                "plugin_name": default_plugin or "unknown",
                "invoked": False,
                "prompt_used": "",
                "image_url": None,
                "image_path": None,
                "elapsed_ms": elapsed_ms,
                "message": f"T2I hook failed: {exc}",
                "extra": {"error_type": type(exc).__name__},
            }

    @staticmethod
    def _validate_chat_inputs(
        *,
        message: str,
        session_id: str | None,
        params: dict[str, Any] | None,
        images: list[str] | None,
    ) -> list[str]:
        if session_id is not None and not isinstance(session_id, str):
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a string.",
                http_status=400,
            )
        if not isinstance(message, str) or not message.strip():
            raise ChatError(
                "CHAT_INVALID_MESSAGE",
                "Field 'message' must be a non-empty string.",
                http_status=400,
            )
        normalized_images: list[str] = []
        if images is not None:
            if not isinstance(images, list):
                raise ChatError(
                    "CHAT_INVALID_IMAGES",
                    "Optional 'images' must be a list of image data URLs or base64 strings.",
                    http_status=400,
                )
            for img in images:
                if not isinstance(img, str) or not img.strip():
                    raise ChatError(
                        "CHAT_INVALID_IMAGES",
                        "Each image entry must be a non-empty string.",
                        http_status=400,
                    )
                cleaned_img = img.strip()
                if not (
                    cleaned_img.startswith("data:image/")
                    or cleaned_img.startswith("http://")
                    or cleaned_img.startswith("https://")
                ):
                    cleaned_img = f"data:image/png;base64,{cleaned_img}"
                normalized_images.append(cleaned_img)
        if params is not None:
            if not isinstance(params, Mapping):
                raise ChatError(
                    "CHAT_INVALID_ARGUMENT",
                    "Optional 'params' must be a mapping.",
                    http_status=400,
                )
            for float_key in ("temperature", "top_p", "min_p", "repeat_penalty", "presence_penalty"):
                if float_key in params:
                    val = params[float_key]
                    if (
                        isinstance(val, bool)
                        or not isinstance(val, (int, float))
                        or not math.isfinite(float(val))
                    ):
                        raise ChatError(
                            "CHAT_INVALID_ARGUMENT",
                            f"Parameter '{float_key}' must be a finite number.",
                            http_status=400,
                        )
            if "max_tokens" in params:
                max_tok = params["max_tokens"]
                if isinstance(max_tok, bool) or not isinstance(max_tok, int) or max_tok < 1:
                    raise ChatError(
                        "CHAT_INVALID_ARGUMENT",
                        "Parameter 'max_tokens' must be an integer >= 1.",
                        http_status=400,
                    )
        return normalized_images

    @staticmethod
    def _build_roleplay_director_note(skill_cfg: dict[str, Any]) -> str:
        """Assemble optional deep-roleplay guidance (immersion, persona, scenario, author's note)."""
        sections: list[str] = []
        if bool(skill_cfg.get("immersion_mode", False)):
            sections.append(_IMMERSION_DIRECTIVE)
        user_name = str(skill_cfg.get("user_name", "") or "").strip()
        user_persona = str(skill_cfg.get("user_persona", "") or "").strip()
        if user_persona:
            label = f"对话对象（{user_name}）" if user_name else "对话对象"
            sections.append(f"【{label}设定】\n{user_persona}")
        elif user_name and sections:
            sections.append(f"【对话对象称呼】对方名为「{user_name}」。")
        scenario = str(skill_cfg.get("scenario", "") or "").strip()
        if scenario:
            sections.append(f"【当前场景与背景】\n{scenario}")
        authors_note = str(skill_cfg.get("authors_note", "") or "").strip()
        if authors_note:
            sections.append(f"【导演提示 / Scene Note】\n{authors_note}")
        return "\n\n".join(sections).strip()

    def _build_messages_and_params(
        self,
        *,
        session: ChatSession,
        skill: SkillLoadResult | None,
        cfg: dict[str, Any],
        user_message: str,
        normalized_images: list[str],
        params: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        llama_cfg = cfg.get("llama", {})
        skill_cfg = cfg.get("skill", {})
        max_history_turns = int(llama_cfg.get("max_history_turns", 20))

        messages_to_send: list[dict[str, Any]] = []
        if skill is not None and skill.system_prompt:
            messages_to_send.append({"role": "system", "content": skill.system_prompt})

        if max_history_turns > 0 and session.messages:
            window = session.messages[-(max_history_turns * 2) :]
        else:
            window = []
        history_entries: list[dict[str, Any]] = [
            {"role": item["role"], "content": item["content"]} for item in window
        ]

        if normalized_images:
            user_content: str | list[dict[str, Any]] = [
                {"type": "text", "text": user_message},
                *[
                    {"type": "image_url", "image_url": {"url": img_url}}
                    for img_url in normalized_images
                ],
            ]
        else:
            user_content = user_message
        history_entries.append({"role": "user", "content": user_content})

        director_note = self._build_roleplay_director_note(skill_cfg)
        if director_note:
            depth = int(skill_cfg.get("authors_note_depth", 2))
            depth = max(0, min(depth, len(history_entries)))
            insert_pos = max(0, len(history_entries) - depth)
            history_entries.insert(insert_pos, {"role": "system", "content": director_note})

        messages_to_send.extend(history_entries)

        effective_params: dict[str, Any] = {
            "temperature": llama_cfg.get("temperature", 0.8),
            "top_p": llama_cfg.get("top_p", 0.9),
            "max_tokens": llama_cfg.get("max_tokens", 512),
        }
        for opt_key in ("min_p", "repeat_penalty", "presence_penalty", "cache_prompt"):
            if opt_key in llama_cfg:
                effective_params[opt_key] = llama_cfg[opt_key]
        if params is not None:
            effective_params.update(params)

        return messages_to_send, effective_params

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        params: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process a single chat turn with prompt injection, history window, multimodal images, and T2I dispatch."""
        normalized_images = self._validate_chat_inputs(
            message=message,
            session_id=session_id,
            params=params,
            images=images,
        )

        sid = session_id.strip() if (session_id is not None and session_id.strip()) else uuid.uuid4().hex
        session = self._get_or_create_session(sid)

        with session.lock:
            cfg = self._config_store.load()
            t2i_cfg = cfg.get("t2i", {})

            with self._state_lock:
                skill = self._current_skill

            messages_to_send, effective_params = self._build_messages_and_params(
                session=session,
                skill=skill,
                cfg=cfg,
                user_message=message,
                normalized_images=normalized_images,
                params=params,
            )

            engine = self._get_or_refresh_engine(cfg)
            raw_res = engine.chat_completion(messages_to_send, effective_params)
            reply, usage = _extract_reply_and_usage(raw_res)

            session.messages.append({"role": "user", "content": message})
            session.messages.append({"role": "assistant", "content": reply})
            session.turn_index += 1
            if skill is not None:
                session.character_name = skill.character_name
            if not session.title:
                session.title = message[:28].replace("\n", " ").strip()
            session.updated_at = self._now_iso()
            self._persist_session(session)

            turn_index = session.turn_index
            messages_count = len(session.messages)

            t2i_hook = self._dispatch_t2i_hook(
                session_id=sid,
                turn_index=turn_index,
                skill=skill,
                user_message=message,
                assistant_reply=reply,
                session_messages=session.messages,
                messages_count=messages_count,
                t2i_cfg=t2i_cfg,
            )

        return {
            "session_id": sid,
            "turn_index": turn_index,
            "reply": reply,
            "usage": usage,
            "t2i_hook": t2i_hook,
            "messages_count": messages_count,
        }

    def chat_stream(
        self,
        message: str,
        session_id: str | None = None,
        params: dict[str, Any] | None = None,
        images: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        """Stream a single chat turn token-by-token as structured SSE event dicts."""
        normalized_images = self._validate_chat_inputs(
            message=message,
            session_id=session_id,
            params=params,
            images=images,
        )
        sid = session_id.strip() if (session_id is not None and session_id.strip()) else uuid.uuid4().hex
        session = self._get_or_create_session(sid)

        with session.lock:
            cfg = self._config_store.load()
            t2i_cfg = cfg.get("t2i", {})
            with self._state_lock:
                skill = self._current_skill

            messages_to_send, effective_params = self._build_messages_and_params(
                session=session,
                skill=skill,
                cfg=cfg,
                user_message=message,
                normalized_images=normalized_images,
                params=params,
            )
            next_turn = session.turn_index + 1
            yield {
                "type": "meta",
                "session_id": sid,
                "turn_index": next_turn,
                "character_name": skill.character_name if skill is not None else "",
            }

            engine = self._get_or_refresh_engine(cfg)
            started = time.perf_counter()
            collected: list[str] = []
            usage: dict[str, Any] = {}
            timings: dict[str, Any] = {}
            aborted = False

            if hasattr(engine, "chat_completion_stream") and callable(engine.chat_completion_stream):
                for ev in engine.chat_completion_stream(
                    messages_to_send,
                    effective_params,
                    cancel_event=cancel_event,
                ):
                    ev_type = ev.get("type")
                    if ev_type == "delta":
                        chunk = str(ev.get("content") or "")
                        if chunk:
                            collected.append(chunk)
                            yield {"type": "delta", "content": chunk}
                    elif ev_type == "done":
                        if not collected and ev.get("reply"):
                            collected.append(str(ev["reply"]))
                        if isinstance(ev.get("usage"), dict):
                            usage = dict(ev["usage"])
                        if isinstance(ev.get("timings"), dict):
                            timings = dict(ev["timings"])
                        aborted = bool(ev.get("aborted", False))
            else:
                raw_res = engine.chat_completion(messages_to_send, effective_params)
                reply_fallback, usage = _extract_reply_and_usage(raw_res)
                collected.append(reply_fallback)
                yield {"type": "delta", "content": reply_fallback}

            elapsed_sec = max(0.001, time.perf_counter() - started)
            reply = "".join(collected)
            if not reply and aborted:
                yield {
                    "type": "done",
                    "session_id": sid,
                    "turn_index": session.turn_index,
                    "reply": "",
                    "usage": usage,
                    "timings": timings,
                    "elapsed_ms": round(elapsed_sec * 1000.0, 1),
                    "t2i_hook": _disabled_t2i_hook(t2i_cfg),
                    "messages_count": len(session.messages),
                    "aborted": True,
                }
                return

            session.messages.append({"role": "user", "content": message})
            session.messages.append({"role": "assistant", "content": reply})
            session.turn_index += 1
            if skill is not None:
                session.character_name = skill.character_name
            if not session.title:
                session.title = message[:28].replace("\n", " ").strip()
            session.updated_at = self._now_iso()
            self._persist_session(session)

            turn_index = session.turn_index
            messages_count = len(session.messages)

            t2i_hook = (
                _disabled_t2i_hook(t2i_cfg)
                if aborted
                else self._dispatch_t2i_hook(
                    session_id=sid,
                    turn_index=turn_index,
                    skill=skill,
                    user_message=message,
                    assistant_reply=reply,
                    session_messages=session.messages,
                    messages_count=messages_count,
                    t2i_cfg=t2i_cfg,
                )
            )

            yield {
                "type": "done",
                "session_id": sid,
                "turn_index": turn_index,
                "reply": reply,
                "usage": usage,
                "timings": timings,
                "elapsed_ms": round(elapsed_sec * 1000.0, 1),
                "t2i_hook": t2i_hook,
                "messages_count": messages_count,
                "aborted": aborted,
            }

    def regenerate_stream(
        self,
        session_id: str,
        params: dict[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        """Pop the last assistant reply and regenerate a fresh response for the last user message."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ChatError(
                "CHAT_INVALID_SESSION_ID",
                "Field 'session_id' must be a non-empty string.",
                http_status=400,
            )
        sid = session_id.strip()
        with self._state_lock:
            session = self._sessions.get(sid)
        if session is None or not session.messages:
            raise ChatError(
                "CHAT_NOTHING_TO_REGENERATE",
                "No conversation history available to regenerate.",
                http_status=400,
            )

        with session.lock:
            if session.messages and session.messages[-1]["role"] == "assistant":
                session.messages.pop()
                session.turn_index = max(0, session.turn_index - 1)
            if not session.messages or session.messages[-1]["role"] != "user":
                raise ChatError(
                    "CHAT_NOTHING_TO_REGENERATE",
                    "Last message must be a user message to regenerate.",
                    http_status=400,
                )
            last_user_msg = session.messages.pop()["content"]
            session.updated_at = self._now_iso()
            self._persist_session(session)

        yield from self.chat_stream(
            message=last_user_msg,
            session_id=sid,
            params=params,
            cancel_event=cancel_event,
        )

    def send_turn(
        self,
        message: str,
        session_id: str | None = None,
        params: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Alias for chat() preserving M2/M4 interface compatibility."""
        return self.chat(message=message, session_id=session_id, params=params, images=images)
