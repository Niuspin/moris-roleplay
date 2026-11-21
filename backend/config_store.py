"""Atomic JSON configuration store and schema validation."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VALID_ENGINE_MODES = frozenset({"managed_server", "external_server", "cli"})

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 7860,
        "open_browser": True,
    },
    "skill": {
        "last_path": "",
        "exclude_meta_docs": False,
        "max_prompt_chars": 0,
        "library_dirs": [
            "skills",
        ],
        "user_name": "旅者",
        "user_persona": "",
        "scenario": "",
        "authors_note": "",
        "authors_note_depth": 2,
        "immersion_mode": False,
    },
    "llama": {
        "engine_mode": "managed_server",
        "engine_dir": "engines",
        "models_dir": "models",
        "llama_cpp_path": "",
        "model_path": "",
        "mmproj_path": "",
        "multimodal_enabled": True,
        "server_url": "http://127.0.0.1:8080",
        "host": "127.0.0.1",
        "port": 8080,
        "ctx_size": 16384,
        "n_gpu_layers": -1,
        "startup_timeout_sec": 60,
        "request_timeout_sec": 120,
        "temperature": 0.8,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.12,
        "presence_penalty": 0.1,
        "cache_prompt": True,
        "max_tokens": 512,
        "max_history_turns": 20,
    },
    "t2i": {
        "enabled": False,
        "active_plugin": "noop",
        "context_window_turns": 4,
        "plugins_dir": "plugins/t2i",
        "plugin_settings": {},
    },
}


def default_config() -> dict[str, Any]:
    """Return a deep copy of the default configuration schema."""
    return copy.deepcopy(DEFAULT_CONFIG)


class ConfigError(Exception):
    """Structured configuration error compatible with HTTP JSON error responses."""

    def __init__(
        self,
        code: str,
        message: str,
        detail: str = "",
        field: str = "",
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.field = field
        self.http_status = http_status

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        if self.field:
            payload["field"] = self.field
        return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override onto a deep copy of base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be a boolean.",
            detail=f"Expected bool, got {type(value).__name__}.",
            field=field,
            http_status=400,
        )
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be an integer.",
            detail=f"Expected int (not bool), got {type(value).__name__}.",
            field=field,
            http_status=400,
        )
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be a finite number.",
            detail=f"Expected int or float (not bool), got {type(value).__name__}.",
            field=field,
            http_status=400,
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be a finite number.",
            detail="NaN and Infinity are not permitted.",
            field=field,
            http_status=400,
        )
    return fval


def _require_str(value: Any, field: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be a string.",
            detail=f"Expected str, got {type(value).__name__}.",
            field=field,
            http_status=400,
        )
    if non_empty and not value.strip():
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message=f"Field '{field}' must not be empty.",
            field=field,
            http_status=400,
        )
    return value


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message=f"Field '{field}' must be an object.",
            detail=f"Expected dict, got {type(value).__name__}.",
            field=field,
            http_status=400,
        )
    return value


_OPTIONAL_EXTENDED_KEYS: dict[str, frozenset[str]] = {
    "skill": frozenset(
        {
            "library_dirs",
            "user_name",
            "user_persona",
            "scenario",
            "authors_note",
            "authors_note_depth",
            "immersion_mode",
        }
    ),
    "llama": frozenset(
        {
            "engine_dir",
            "models_dir",
            "mmproj_path",
            "multimodal_enabled",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "cache_prompt",
        }
    ),
}


def _validate_required_structure(data: dict[str, Any]) -> None:
    """Verify all top-level sections and required schema keys exist."""
    for section, defaults in DEFAULT_CONFIG.items():
        if section not in data:
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message=f"Missing required configuration section '{section}'.",
                field=section,
                http_status=400,
            )
        section_val = _require_dict(data[section], section)
        optional_for_sec = _OPTIONAL_EXTENDED_KEYS.get(section, frozenset())
        for key, default_val in defaults.items():
            if key not in section_val:
                if key in optional_for_sec:
                    section_val[key] = copy.deepcopy(default_val)
                    continue
                full_field = f"{section}.{key}"
                raise ConfigError(
                    code="CONFIG_VALIDATION_ERROR",
                    message=f"Missing required configuration key '{full_field}'.",
                    field=full_field,
                    http_status=400,
                )


def _validate_config(data: Any) -> dict[str, Any]:
    """Validate configuration structure, field types, and value ranges."""
    root = _require_dict(data, "root")
    _validate_required_structure(root)

    # server
    server = root["server"]
    _require_str(server["host"], "server.host", non_empty=True)
    server_port = _require_int(server["port"], "server.port")
    if not (1 <= server_port <= 65535):
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'server.port' must be between 1 and 65535.",
            detail=f"Got {server_port}.",
            field="server.port",
            http_status=400,
        )
    _require_bool(server["open_browser"], "server.open_browser")

    # skill
    skill = root["skill"]
    _require_str(skill["last_path"], "skill.last_path", non_empty=False)
    _require_bool(skill["exclude_meta_docs"], "skill.exclude_meta_docs")
    max_prompt_chars = _require_int(skill["max_prompt_chars"], "skill.max_prompt_chars")
    if max_prompt_chars < 0:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'skill.max_prompt_chars' must be >= 0.",
            detail=f"Got {max_prompt_chars}.",
            field="skill.max_prompt_chars",
            http_status=400,
        )
    if "library_dirs" in skill:
        lib_dirs = skill["library_dirs"]
        if not isinstance(lib_dirs, list) or any(not isinstance(item, str) for item in lib_dirs):
            raise ConfigError(
                code="CONFIG_INVALID_TYPE",
                message="Field 'skill.library_dirs' must be a list of strings.",
                field="skill.library_dirs",
                http_status=400,
            )
    if "user_name" in skill:
        _require_str(skill["user_name"], "skill.user_name", non_empty=False)
    if "user_persona" in skill:
        _require_str(skill["user_persona"], "skill.user_persona", non_empty=False)
    if "scenario" in skill:
        _require_str(skill["scenario"], "skill.scenario", non_empty=False)
    if "authors_note" in skill:
        _require_str(skill["authors_note"], "skill.authors_note", non_empty=False)
    if "authors_note_depth" in skill:
        an_depth = _require_int(skill["authors_note_depth"], "skill.authors_note_depth")
        if an_depth < 0 or an_depth > 20:
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'skill.authors_note_depth' must be between 0 and 20.",
                detail=f"Got {an_depth}.",
                field="skill.authors_note_depth",
                http_status=400,
            )
    if "immersion_mode" in skill:
        _require_bool(skill["immersion_mode"], "skill.immersion_mode")

    # llama
    llama = root["llama"]
    engine_mode = _require_str(llama["engine_mode"], "llama.engine_mode", non_empty=True)
    if engine_mode not in VALID_ENGINE_MODES:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message=(
                "Field 'llama.engine_mode' must be one of: "
                + ", ".join(sorted(VALID_ENGINE_MODES))
                + "."
            ),
            detail=f"Got {engine_mode!r}.",
            field="llama.engine_mode",
            http_status=400,
        )
    if "engine_dir" in llama:
        _require_str(llama["engine_dir"], "llama.engine_dir", non_empty=False)
    if "models_dir" in llama:
        _require_str(llama["models_dir"], "llama.models_dir", non_empty=False)
    _require_str(llama["llama_cpp_path"], "llama.llama_cpp_path", non_empty=False)
    _require_str(llama["model_path"], "llama.model_path", non_empty=False)
    if "mmproj_path" in llama:
        _require_str(llama["mmproj_path"], "llama.mmproj_path", non_empty=False)
    if "multimodal_enabled" in llama:
        _require_bool(llama["multimodal_enabled"], "llama.multimodal_enabled")
    server_url = _require_str(llama["server_url"], "llama.server_url", non_empty=False)
    if server_url != "":
        if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in server_url):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' must not contain whitespace or control characters.",
                detail=f"Got {server_url!r}.",
                field="llama.server_url",
                http_status=400,
            )
        try:
            parsed = urlparse(server_url)
            url_scheme = parsed.scheme
            url_netloc = parsed.netloc
            url_hostname = parsed.hostname
            url_port = parsed.port
        except ValueError as exc:
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' must be empty or a valid http/https URL.",
                detail=f"Got {server_url!r}: {exc}",
                field="llama.server_url",
                http_status=400,
            ) from exc
        if url_scheme not in {"http", "https"} or not url_netloc:
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' must be empty or a valid http/https URL.",
                detail=f"Got {server_url!r}.",
                field="llama.server_url",
                http_status=400,
            )
        hostport = url_netloc.rsplit("@", 1)[-1]
        if hostport.endswith(":"):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' must not end with an empty port.",
                detail=f"Got {server_url!r}.",
                field="llama.server_url",
                http_status=400,
            )
        if hostport.startswith("["):
            _, sep, after_bracket = hostport.partition("]")
            if not sep or (after_bracket and not after_bracket.startswith(":")):
                raise ConfigError(
                    code="CONFIG_VALIDATION_ERROR",
                    message="Field 'llama.server_url' must be empty or a valid http/https URL.",
                    detail=f"Got {server_url!r}.",
                    field="llama.server_url",
                    http_status=400,
                )
        if not url_hostname or any(ch.isspace() for ch in url_hostname):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' must contain a valid hostname.",
                detail=f"Got {server_url!r}.",
                field="llama.server_url",
                http_status=400,
            )
        if url_port is not None and not (1 <= url_port <= 65535):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.server_url' port must be between 1 and 65535.",
                detail=f"Got {url_port}.",
                field="llama.server_url",
                http_status=400,
            )
    _require_str(llama["host"], "llama.host", non_empty=True)
    llama_port = _require_int(llama["port"], "llama.port")
    if not (1 <= llama_port <= 65535):
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.port' must be between 1 and 65535.",
            detail=f"Got {llama_port}.",
            field="llama.port",
            http_status=400,
        )
    ctx_size = _require_int(llama["ctx_size"], "llama.ctx_size")
    if ctx_size < 1:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.ctx_size' must be >= 1.",
            detail=f"Got {ctx_size}.",
            field="llama.ctx_size",
            http_status=400,
        )
    n_gpu_layers = _require_int(llama["n_gpu_layers"], "llama.n_gpu_layers")
    if n_gpu_layers < -1:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.n_gpu_layers' must be >= -1.",
            detail=f"Got {n_gpu_layers}.",
            field="llama.n_gpu_layers",
            http_status=400,
        )
    startup_timeout = _require_number(
        llama["startup_timeout_sec"], "llama.startup_timeout_sec"
    )
    if startup_timeout <= 0:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.startup_timeout_sec' must be > 0.",
            detail=f"Got {startup_timeout}.",
            field="llama.startup_timeout_sec",
            http_status=400,
        )
    request_timeout = _require_number(
        llama["request_timeout_sec"], "llama.request_timeout_sec"
    )
    if request_timeout <= 0:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.request_timeout_sec' must be > 0.",
            detail=f"Got {request_timeout}.",
            field="llama.request_timeout_sec",
            http_status=400,
        )
    temperature = _require_number(llama["temperature"], "llama.temperature")
    if not (0.0 <= temperature <= 2.0):
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.temperature' must be between 0.0 and 2.0.",
            detail=f"Got {temperature}.",
            field="llama.temperature",
            http_status=400,
        )
    top_p = _require_number(llama["top_p"], "llama.top_p")
    if not (0.0 <= top_p <= 1.0):
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.top_p' must be between 0.0 and 1.0.",
            detail=f"Got {top_p}.",
            field="llama.top_p",
            http_status=400,
        )
    if "min_p" in llama:
        min_p = _require_number(llama["min_p"], "llama.min_p")
        if not (0.0 <= min_p <= 1.0):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.min_p' must be between 0.0 and 1.0.",
                detail=f"Got {min_p}.",
                field="llama.min_p",
                http_status=400,
            )
    if "repeat_penalty" in llama:
        repeat_penalty = _require_number(llama["repeat_penalty"], "llama.repeat_penalty")
        if not (0.5 <= repeat_penalty <= 2.0):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.repeat_penalty' must be between 0.5 and 2.0.",
                detail=f"Got {repeat_penalty}.",
                field="llama.repeat_penalty",
                http_status=400,
            )
    if "presence_penalty" in llama:
        presence_penalty = _require_number(llama["presence_penalty"], "llama.presence_penalty")
        if not (-2.0 <= presence_penalty <= 2.0):
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Field 'llama.presence_penalty' must be between -2.0 and 2.0.",
                detail=f"Got {presence_penalty}.",
                field="llama.presence_penalty",
                http_status=400,
            )
    if "cache_prompt" in llama:
        _require_bool(llama["cache_prompt"], "llama.cache_prompt")
    max_tokens = _require_int(llama["max_tokens"], "llama.max_tokens")
    if max_tokens < 1:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.max_tokens' must be >= 1.",
            detail=f"Got {max_tokens}.",
            field="llama.max_tokens",
            http_status=400,
        )
    max_history_turns = _require_int(
        llama["max_history_turns"], "llama.max_history_turns"
    )
    if max_history_turns < 1:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 'llama.max_history_turns' must be >= 1.",
            detail=f"Got {max_history_turns}.",
            field="llama.max_history_turns",
            http_status=400,
        )

    # t2i
    t2i = root["t2i"]
    _require_bool(t2i["enabled"], "t2i.enabled")
    _require_str(t2i["active_plugin"], "t2i.active_plugin", non_empty=True)
    context_window_turns = _require_int(
        t2i["context_window_turns"], "t2i.context_window_turns"
    )
    if context_window_turns < 1:
        raise ConfigError(
            code="CONFIG_VALIDATION_ERROR",
            message="Field 't2i.context_window_turns' must be >= 1.",
            detail=f"Got {context_window_turns}.",
            field="t2i.context_window_turns",
            http_status=400,
        )
    _require_str(t2i["plugins_dir"], "t2i.plugins_dir", non_empty=True)
    plugin_settings = _require_dict(t2i["plugin_settings"], "t2i.plugin_settings")
    try:
        json.dumps(plugin_settings, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message="Field 't2i.plugin_settings' must be JSON-serializable.",
            detail=str(exc),
            field="t2i.plugin_settings",
            http_status=400,
        ) from exc

    # Verify whole root is JSON-serializable without NaN/Inf
    try:
        json.dumps(root, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            code="CONFIG_INVALID_TYPE",
            message="Configuration contains non-JSON-serializable values.",
            detail=str(exc),
            http_status=400,
        ) from exc

    return root


class ConfigStore:
    """Thread-safe atomic JSON configuration store."""

    def __init__(
        self,
        config_path: Path | str,
        example_path: Path | str | None = None,
    ) -> None:
        if isinstance(config_path, str) and not config_path.strip():
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Configuration path must not be empty.",
                field="config_path",
                http_status=400,
            )
        if isinstance(example_path, str) and not example_path.strip():
            raise ConfigError(
                code="CONFIG_VALIDATION_ERROR",
                message="Example configuration path must not be empty.",
                field="example_path",
                http_status=400,
            )
        self.config_path = Path(config_path)
        self.example_path = Path(example_path) if example_path is not None else None
        self._lock = threading.RLock()

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        if path.is_dir():
            raise ConfigError(
                code="CONFIG_IO_ERROR",
                message="Configuration path points to a directory.",
                detail=str(path),
                http_status=500,
            )
        try:
            raw_text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ConfigError(
                code="CONFIG_INVALID_JSON",
                message="Configuration file is not valid UTF-8.",
                detail=str(exc),
                http_status=400,
            ) from exc
        except OSError as exc:
            raise ConfigError(
                code="CONFIG_IO_ERROR",
                message="Failed to read configuration file.",
                detail=str(exc),
                http_status=500,
            ) from exc

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                code="CONFIG_INVALID_JSON",
                message="Configuration file contains invalid JSON.",
                detail=str(exc),
                http_status=400,
            ) from exc

        if not isinstance(parsed, dict):
            raise ConfigError(
                code="CONFIG_INVALID_TYPE",
                message="Configuration root must be a JSON object.",
                detail=f"Expected dict, got {type(parsed).__name__}.",
                field="root",
                http_status=400,
            )
        return parsed

    def _build_initial_config(self) -> dict[str, Any]:
        base = default_config()
        if self.example_path is not None and self.example_path.exists():
            seed = self._read_json_file(self.example_path)
            base = _deep_merge(base, seed)
        _validate_config(base)
        return base

    def _write_atomic(self, data: dict[str, Any]) -> None:
        try:
            serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                code="CONFIG_INVALID_TYPE",
                message="Configuration could not be serialized to JSON.",
                detail=str(exc),
                http_status=400,
            ) from exc

        parent_dir = self.config_path.parent
        tmp_path: str | None = None
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=parent_dir,
                prefix=f".{self.config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(serialized)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, self.config_path)
            tmp_path = None
        except ConfigError:
            raise
        except OSError as exc:
            raise ConfigError(
                code="CONFIG_IO_ERROR",
                message="Failed to write configuration file atomically.",
                detail=str(exc),
                http_status=500,
            ) from exc
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _read_effective_unlocked(self) -> tuple[dict[str, Any], bool]:
        """Return (validated_config, existed_on_disk) without writing to disk."""
        if not self.config_path.exists():
            return self._build_initial_config(), False
        raw = self._read_json_file(self.config_path)
        merged = _deep_merge(DEFAULT_CONFIG, raw)
        _validate_config(merged)
        return merged, True

    def load(self) -> dict[str, Any]:
        """Load configuration from disk, creating from defaults/example if missing."""
        with self._lock:
            cfg, existed = self._read_effective_unlocked()
            if not existed:
                self._write_atomic(cfg)
            return copy.deepcopy(cfg)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate full configuration and write it atomically to disk."""
        with self._lock:
            _require_dict(data, "root")
            candidate = copy.deepcopy(data)
            _validate_config(candidate)
            self._write_atomic(candidate)
            return copy.deepcopy(candidate)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge patch into current configuration, validate, and save atomically."""
        with self._lock:
            _require_dict(patch, "patch")
            current, _ = self._read_effective_unlocked()
            candidate = _deep_merge(current, patch)
            _validate_config(candidate)
            self._write_atomic(candidate)
            return copy.deepcopy(candidate)
