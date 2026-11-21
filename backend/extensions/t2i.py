"""Post-turn scene Text-to-Image (T2I) extension interface, registry, and hook dispatcher."""

from __future__ import annotations

import copy
import importlib.util
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PLUGINS_DIR = REPO_ROOT / "plugins" / "t2i"

T2IStatus = Literal["disabled", "skipped", "stub_ok", "generated", "error"]


@dataclass(frozen=True, slots=True)
class SceneContext:
    """Immutable post-turn conversation scene snapshot passed to T2I plugins."""

    session_id: str
    turn_index: int
    timestamp_utc: str
    character_name: str
    skill_source: str
    user_message: str
    assistant_reply: str
    recent_messages: tuple[dict[str, str], ...]
    scene_prompt_hint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "timestamp_utc": self.timestamp_utc,
            "character_name": self.character_name,
            "skill_source": self.skill_source,
            "user_message": self.user_message,
            "assistant_reply": self.assistant_reply,
            "recent_messages": [dict(m) for m in self.recent_messages],
            "scene_prompt_hint": self.scene_prompt_hint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class T2IResult:
    """Normalized result returned by a T2I plugin or T2IHookDispatcher."""

    status: str
    plugin_name: str
    invoked: bool
    prompt_used: str = ""
    image_url: str | None = None
    image_path: str | None = None
    elapsed_ms: float = 0.0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class T2IPlugin(Protocol):
    """Protocol implemented by post-turn Text-to-Image plugins."""

    name: str
    description: str

    def on_turn_completed(
        self,
        context: SceneContext,
        plugin_settings: dict[str, Any],
    ) -> T2IResult:
        """Handle a completed dialogue turn and return T2IResult metadata."""


class NoopT2IPlugin:
    """Built-in default no-op plugin that performs no external or model call."""

    name: str = "noop"
    description: str = "Built-in no-op plugin (does not invoke any text-to-image model)."

    def on_turn_completed(
        self,
        context: SceneContext,
        plugin_settings: dict[str, Any],
    ) -> T2IResult:
        return T2IResult(
            status="skipped",
            plugin_name=self.name,
            invoked=False,
            prompt_used="",
            image_url=None,
            image_path=None,
            elapsed_ms=0.0,
            message="No-op plugin active; image generation skipped.",
            extra={},
        )


def coerce_t2i_result(
    raw: Any,
    *,
    default_plugin_name: str,
    measured_elapsed_ms: float,
) -> T2IResult:
    """Coerce a T2IResult, duck-typed plugin result object, or dict into a T2IResult."""
    if isinstance(raw, T2IResult):
        elapsed = raw.elapsed_ms if raw.elapsed_ms > 0 else round(measured_elapsed_ms, 3)
        return T2IResult(
            status=str(raw.status or "stub_ok"),
            plugin_name=str(raw.plugin_name or default_plugin_name),
            invoked=bool(raw.invoked),
            prompt_used=str(raw.prompt_used or ""),
            image_url=raw.image_url,
            image_path=raw.image_path,
            elapsed_ms=float(elapsed),
            message=str(raw.message or ""),
            extra=dict(raw.extra) if isinstance(raw.extra, Mapping) else {},
        )

    if isinstance(raw, Mapping):
        data = dict(raw)
    elif hasattr(raw, "to_dict") and callable(raw.to_dict):
        converted = raw.to_dict()
        data = dict(converted) if isinstance(converted, Mapping) else {}
    elif is_dataclass(raw) and not isinstance(raw, type):
        data = asdict(raw)
    else:
        data = {
            "status": getattr(raw, "status", "stub_ok"),
            "plugin_name": getattr(raw, "plugin_name", default_plugin_name),
            "invoked": getattr(raw, "invoked", True),
            "prompt_used": getattr(raw, "prompt_used", ""),
            "image_url": getattr(raw, "image_url", None),
            "image_path": getattr(raw, "image_path", None),
            "elapsed_ms": getattr(raw, "elapsed_ms", measured_elapsed_ms),
            "message": getattr(raw, "message", ""),
            "extra": getattr(raw, "extra", {}),
        }

    raw_elapsed = data.get("elapsed_ms")
    if isinstance(raw_elapsed, (int, float)) and float(raw_elapsed) > 0:
        elapsed_ms = float(raw_elapsed)
    else:
        elapsed_ms = round(measured_elapsed_ms, 3)

    extra_raw = data.get("extra")
    extra_dict = dict(extra_raw) if isinstance(extra_raw, Mapping) else {}

    return T2IResult(
        status=str(data.get("status") or "stub_ok"),
        plugin_name=str(data.get("plugin_name") or default_plugin_name),
        invoked=bool(data.get("invoked", True)),
        prompt_used=str(data.get("prompt_used") or ""),
        image_url=data.get("image_url"),
        image_path=data.get("image_path"),
        elapsed_ms=elapsed_ms,
        message=str(data.get("message") or ""),
        extra=extra_dict,
    )


class T2IRegistry:
    """Thread-safe registry and directory loader for post-turn T2I plugins."""

    def __init__(self, *, register_builtins: bool = True) -> None:
        self._lock = threading.RLock()
        self._plugins: dict[str, Any] = {}
        self._discovery_errors: dict[str, str] = {}
        if register_builtins:
            self.register(NoopT2IPlugin())
            if DEFAULT_PLUGINS_DIR.is_dir():
                self.discover_directory(DEFAULT_PLUGINS_DIR)

    def register(self, plugin: Any) -> None:
        """Validate and register a T2IPlugin instance."""
        name = getattr(plugin, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("T2I plugin must define a non-empty string 'name' attribute.")
        hook = getattr(plugin, "on_turn_completed", None)
        if not callable(hook):
            raise ValueError(
                f"T2I plugin '{name}' must define a callable 'on_turn_completed(context, plugin_settings)'."
            )
        clean_name = name.strip()
        with self._lock:
            self._plugins[clean_name] = plugin

    def get(self, name: str) -> Any | None:
        """Return a registered plugin by name, or None if not found."""
        if not isinstance(name, str):
            return None
        with self._lock:
            return self._plugins.get(name.strip())

    def list_plugins(self) -> list[dict[str, str]]:
        """Return metadata for all registered plugins with 'noop' first."""
        with self._lock:
            items = list(self._plugins.items())
        ordered: list[tuple[str, Any]] = []
        for k, v in items:
            if k == "noop":
                ordered.append((k, v))
        for k, v in sorted(items, key=lambda pair: pair[0]):
            if k != "noop":
                ordered.append((k, v))
        return [
            {
                "name": name,
                "description": str(getattr(plugin, "description", "") or ""),
            }
            for name, plugin in ordered
        ]

    @property
    def discovery_errors(self) -> dict[str, str]:
        """Return a copy of any plugin file errors encountered during directory discovery."""
        with self._lock:
            return dict(self._discovery_errors)

    def discover_directory(self, directory: str | Path) -> list[str]:
        """Discover and register drop-in plugins from ``directory/*.py``.

        Supports two drop-in patterns:
        1. ``def register(registry: T2IRegistry) -> None``
        2. ``PLUGIN = <T2IPlugin instance>``

        Non-``.py`` files and files starting with ``_`` are ignored. Broken plugin files
        are isolated and recorded in ``discovery_errors`` without crashing discovery.
        """
        if not directory:
            return []
        raw_path = Path(directory)
        candidates: list[Path] = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(REPO_ROOT / raw_path)

        target_dir: Path | None = None
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                target_dir = cand.resolve()
                break

        if target_dir is None:
            return []

        discovered: list[str] = []
        try:
            entries = sorted(target_dir.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return []

        for file_path in entries:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != ".py":
                continue
            if file_path.name.startswith("_"):
                continue

            module_name = f"_moris_t2i_plugin_{file_path.stem}_{abs(hash(str(file_path)))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                before_names = set(self._plugins.keys())
                register_fn = getattr(module, "register", None)
                if callable(register_fn):
                    register_fn(self)
                elif hasattr(module, "PLUGIN"):
                    self.register(getattr(module, "PLUGIN"))
                else:
                    continue

                after_names = set(self._plugins.keys())
                new_names = sorted(after_names - before_names)
                if new_names:
                    discovered.extend(new_names)
                elif hasattr(module, "PLUGIN"):
                    p_name = getattr(getattr(module, "PLUGIN"), "name", "")
                    if isinstance(p_name, str) and p_name.strip():
                        discovered.append(p_name.strip())
            except Exception as exc:
                sys.modules.pop(module_name, None)
                with self._lock:
                    self._discovery_errors[str(file_path)] = f"{type(exc).__name__}: {exc}"
                continue

        return discovered


class T2IHookDispatcher:
    """Dispatches post-turn SceneContext to the active T2I plugin with fault isolation."""

    def __init__(
        self,
        registry: T2IRegistry | None = None,
        config_getter: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.registry = registry if registry is not None else T2IRegistry()
        self._config_getter = config_getter

    def _resolve_t2i_config(self, override: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if override is not None:
            if "enabled" in override or "active_plugin" in override or "plugins_dir" in override:
                return dict(override)
            if isinstance(override.get("t2i"), Mapping):
                return dict(override["t2i"])
            return dict(override)

        if self._config_getter is not None:
            full_or_t2i = self._config_getter()
            if isinstance(full_or_t2i, Mapping):
                if isinstance(full_or_t2i.get("t2i"), Mapping):
                    return dict(full_or_t2i["t2i"])
                return dict(full_or_t2i)

        return {
            "enabled": False,
            "active_plugin": "noop",
            "context_window_turns": 4,
            "plugins_dir": "plugins/t2i",
            "plugin_settings": {},
        }

    def refresh_plugins(self, t2i_config: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
        """Discover plugins in the configured ``plugins_dir`` and return available plugins."""
        cfg = self._resolve_t2i_config(t2i_config)
        plugins_dir = str(cfg.get("plugins_dir", "plugins/t2i") or "plugins/t2i").strip()
        if plugins_dir:
            self.registry.discover_directory(plugins_dir)
        return self.registry.list_plugins()

    def status_summary(self, t2i_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return ``/api/extensions/t2i`` status payload."""
        cfg = self._resolve_t2i_config(t2i_config)
        plugins = self.refresh_plugins(cfg)
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "active_plugin": str(cfg.get("active_plugin", "noop") or "noop"),
            "context_window_turns": int(cfg.get("context_window_turns", 4)),
            "plugins_dir": str(cfg.get("plugins_dir", "plugins/t2i") or "plugins/t2i"),
            "plugins": plugins,
        }

    def dispatch(
        self,
        context: SceneContext,
        t2i_config: Mapping[str, Any] | None = None,
    ) -> T2IResult:
        """Run the post-turn T2I hook according to ``t2i_config`` without raising exceptions."""
        cfg = self._resolve_t2i_config(t2i_config)
        enabled = bool(cfg.get("enabled", False))
        active_plugin = str(cfg.get("active_plugin", "noop") or "noop").strip() or "noop"

        if not enabled:
            return T2IResult(
                status="disabled",
                plugin_name=active_plugin,
                invoked=False,
                prompt_used="",
                image_url=None,
                image_path=None,
                elapsed_ms=0.0,
                message="T2I hook is disabled.",
                extra={},
            )

        plugins_dir = str(cfg.get("plugins_dir", "plugins/t2i") or "plugins/t2i").strip()
        if plugins_dir:
            self.registry.discover_directory(plugins_dir)

        if active_plugin == "noop":
            noop_plugin = self.registry.get("noop") or NoopT2IPlugin()
            return noop_plugin.on_turn_completed(context, {})

        plugin = self.registry.get(active_plugin)
        if plugin is None:
            return T2IResult(
                status="error",
                plugin_name=active_plugin,
                invoked=False,
                prompt_used="",
                image_url=None,
                image_path=None,
                elapsed_ms=0.0,
                message=f"Active T2I plugin '{active_plugin}' was not found in registry.",
                extra={"error_code": "PLUGIN_NOT_FOUND"},
            )

        raw_settings = cfg.get("plugin_settings")
        plugin_settings = copy.deepcopy(dict(raw_settings)) if isinstance(raw_settings, Mapping) else {}

        started = time.perf_counter()
        try:
            raw_res = plugin.on_turn_completed(context, plugin_settings)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return coerce_t2i_result(
                raw_res,
                default_plugin_name=active_plugin,
                measured_elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return T2IResult(
                status="error",
                plugin_name=active_plugin,
                invoked=True,
                prompt_used="",
                image_url=None,
                image_path=None,
                elapsed_ms=elapsed_ms,
                message=f"T2I plugin '{active_plugin}' failed: {exc}",
                extra={"error_type": type(exc).__name__},
            )
