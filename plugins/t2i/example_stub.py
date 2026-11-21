"""Reference dry-run Text-to-Image (T2I) stub plugin for Moris Roleplay.

This plugin demonstrates the drop-in T2I extension contract without making any
network request or invoking any external image generation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StubT2IResult:
    """Result structure compatible with backend.extensions.t2i.T2IResult."""

    status: str
    plugin_name: str
    invoked: bool
    prompt_used: str = ""
    image_url: str | None = None
    image_path: str | None = None
    elapsed_ms: float = 0.0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ExampleStubT2IPlugin:
    """Dry-run stub plugin synthesizing a scene prompt from SceneContext."""

    name = "example_stub"
    description = (
        "Reference dry-run T2I stub plugin (synthesizes a scene prompt from "
        "dialogue context without calling an external image model)."
    )

    def on_turn_completed(
        self,
        context: Any,
        plugin_settings: dict[str, Any],
    ) -> StubT2IResult:
        character = getattr(context, "character_name", "") or "character"
        hint = getattr(context, "scene_prompt_hint", "") or ""
        user_msg = getattr(context, "user_message", "") or ""
        reply = getattr(context, "assistant_reply", "") or ""
        style = str((plugin_settings or {}).get("style", "animeillustration")).strip()

        base_scene = hint if hint else f"{character}: {reply[:120]}"
        prompt_used = f"[{style}] {character}, scene: {base_scene} (context: {user_msg[:80]})".strip()

        return StubT2IResult(
            status="stub_ok",
            plugin_name=self.name,
            invoked=True,
            prompt_used=prompt_used,
            image_url=None,
            image_path=None,
            elapsed_ms=0.5,
            message="Dry-run scene prompt synthesized by example_stub (no external T2I model called).",
            extra={
                "turn_index": getattr(context, "turn_index", 0),
                "session_id": getattr(context, "session_id", ""),
                "settings_received": dict(plugin_settings or {}),
            },
        )


PLUGIN = ExampleStubT2IPlugin()


def register(registry: Any) -> None:
    """Register the reference dry-run stub plugin with ``T2IRegistry``."""
    registry.register(PLUGIN)
