"""Extension interfaces for Moris Roleplay."""

from backend.extensions.t2i import (
    NoopT2IPlugin,
    SceneContext,
    T2IHookDispatcher,
    T2IPlugin,
    T2IRegistry,
    T2IResult,
)

__all__ = [
    "NoopT2IPlugin",
    "SceneContext",
    "T2IHookDispatcher",
    "T2IPlugin",
    "T2IRegistry",
    "T2IResult",
]
