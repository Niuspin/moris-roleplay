"""Stateful multi-turn roleplay chat service with skill injection and T2I hook dispatch."""

from __future__ import annotations

import copy
import inspect
import math
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

        self._state_lock = threading.RLock()
        self._engine_lock = threading.RLock()
        self._sessions: dict[str, ChatSession] = {}
        self._current_skill: SkillLoadResult | None = None
        self._engine_fingerprint: tuple[tuple[str, Any], ...] | None = None

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
    # Session & multi-turn chat
    # ------------------------------------------------------------------

    def _get_or_create_session(self, session_id: str) -> ChatSession:
        with self._state_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession(session_id=session_id)
                self._sessions[session_id] = session
            return session

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

    def chat(
        self,
        message: str,
        session_id: str | None = None,
        params: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process a single chat turn with prompt injection, history window, multimodal images, and T2I dispatch."""
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
            for float_key in ("temperature", "top_p"):
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

        sid = session_id.strip() if (session_id is not None and session_id.strip()) else uuid.uuid4().hex
        session = self._get_or_create_session(sid)

        with session.lock:
            cfg = self._config_store.load()
            llama_cfg = cfg.get("llama", {})
            t2i_cfg = cfg.get("t2i", {})
            max_history_turns = int(llama_cfg.get("max_history_turns", 20))

            with self._state_lock:
                skill = self._current_skill

            messages_to_send: list[dict[str, Any]] = []
            if skill is not None and skill.system_prompt:
                messages_to_send.append({"role": "system", "content": skill.system_prompt})

            if max_history_turns > 0 and session.messages:
                window = session.messages[-(max_history_turns * 2) :]
            else:
                window = []
            for item in window:
                messages_to_send.append({"role": item["role"], "content": item["content"]})

            if normalized_images:
                user_content: str | list[dict[str, Any]] = [
                    {"type": "text", "text": message},
                    *[
                        {"type": "image_url", "image_url": {"url": img_url}}
                        for img_url in normalized_images
                    ],
                ]
            else:
                user_content = message
            messages_to_send.append({"role": "user", "content": user_content})

            effective_params: dict[str, Any] = {
                "temperature": llama_cfg.get("temperature", 0.8),
                "top_p": llama_cfg.get("top_p", 0.9),
                "max_tokens": llama_cfg.get("max_tokens", 512),
            }
            if params is not None:
                effective_params.update(params)

            engine = self._get_or_refresh_engine(cfg)
            raw_res = engine.chat_completion(messages_to_send, effective_params)
            reply, usage = _extract_reply_and_usage(raw_res)

            session.messages.append({"role": "user", "content": message})
            session.messages.append({"role": "assistant", "content": reply})
            session.turn_index += 1
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

    def send_turn(
        self,
        message: str,
        session_id: str | None = None,
        params: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """Alias for chat() preserving M2/M4 interface compatibility."""
        return self.chat(message=message, session_id=session_id, params=params, images=images)
