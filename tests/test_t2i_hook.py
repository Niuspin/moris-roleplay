"""Unit tests for M3: Post-Turn Scene T2I Extension Interface (backend.extensions.t2i & plugins.t2i)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from backend.chat_service import ChatService
from backend.config_store import ConfigStore
from backend.extensions.t2i import (
    NoopT2IPlugin,
    SceneContext,
    T2IHookDispatcher,
    T2IPlugin,
    T2IRegistry,
    T2IResult,
)


class _DummyEngine:
    def __init__(self, reply: str = "流萤微微笑了笑。") -> None:
        self._reply = reply

    def status(self) -> dict[str, Any]:
        return {"state": "ready", "mode": "external_server"}

    def ensure_ready(self) -> None:
        return None

    def chat_completion(self, messages: list[dict[str, str]], params: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "reply": self._reply,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def shutdown(self) -> None:
        return None


class T2IHookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="moris_test_t2i_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_dir, ignore_errors=True))

    def _sample_context(self) -> SceneContext:
        return SceneContext(
            session_id="sess-1",
            turn_index=2,
            timestamp_utc="2026-11-20T12:00:00Z",
            character_name="firefly-hsr",
            skill_source="C:/skills/firefly",
            user_message="一起看夜景吧",
            assistant_reply="好呀，秘密基地的星空最美了。",
            recent_messages=(
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，开拓者"},
                {"role": "user", "content": "一起看夜景吧"},
                {"role": "assistant", "content": "好呀，秘密基地的星空最美了。"},
            ),
            scene_prompt_hint="firefly-hsr: 好呀，秘密基地的星空最美了。",
            metadata={"messages_count": 4, "skill_loaded": True},
        )

    def test_builtin_noop_and_example_stub_registration(self) -> None:
        registry = T2IRegistry(register_builtins=True)
        names = {p["name"] for p in registry.list_plugins()}
        self.assertIn("noop", names)
        self.assertIn("example_stub", names)
        self.assertEqual(registry.list_plugins()[0]["name"], "noop")

        noop = registry.get("noop")
        self.assertIsNotNone(noop)
        self.assertIsInstance(noop, T2IPlugin)
        res = noop.on_turn_completed(self._sample_context(), {})
        self.assertEqual(res.status, "skipped")
        self.assertIs(res.invoked, False)

        stub = registry.get("example_stub")
        self.assertIsNotNone(stub)
        stub_res = stub.on_turn_completed(self._sample_context(), {"style": "watercolor"})
        self.assertEqual(stub_res.status, "stub_ok")
        self.assertIs(stub_res.invoked, True)
        self.assertIn("watercolor", stub_res.prompt_used)
        self.assertIn("firefly-hsr", stub_res.prompt_used)

    def test_dispatcher_disabled_and_enabled_modes(self) -> None:
        registry = T2IRegistry(register_builtins=True)
        dispatcher = T2IHookDispatcher(registry=registry)
        ctx = self._sample_context()

        # Disabled by default
        r_disabled = dispatcher.dispatch(ctx, {"enabled": False, "active_plugin": "example_stub"})
        self.assertEqual(r_disabled.status, "disabled")
        self.assertIs(r_disabled.invoked, False)
        self.assertEqual(r_disabled.plugin_name, "example_stub")

        # Enabled with noop
        r_noop = dispatcher.dispatch(ctx, {"enabled": True, "active_plugin": "noop"})
        self.assertEqual(r_noop.status, "skipped")
        self.assertIs(r_noop.invoked, False)
        self.assertEqual(r_noop.plugin_name, "noop")

        # Enabled with example_stub
        r_stub = dispatcher.dispatch(
            ctx,
            {
                "enabled": True,
                "active_plugin": "example_stub",
                "plugin_settings": {"style": "cel-shaded"},
            },
        )
        self.assertEqual(r_stub.status, "stub_ok")
        self.assertIs(r_stub.invoked, True)
        self.assertEqual(r_stub.plugin_name, "example_stub")
        self.assertIn("cel-shaded", r_stub.prompt_used)

        # Enabled with unknown plugin
        r_missing = dispatcher.dispatch(ctx, {"enabled": True, "active_plugin": "unknown_xyz"})
        self.assertEqual(r_missing.status, "error")
        self.assertIs(r_missing.invoked, False)
        self.assertIn("unknown_xyz", r_missing.message)

    def test_directory_discovery_plugin_and_register_and_fault_isolation(self) -> None:
        p_dir = self.tmp_dir / "custom_plugins"
        p_dir.mkdir(parents=True)

        # 1. PLUGIN attribute pattern
        (p_dir / "attr_plugin.py").write_text(
            "class _P:\n"
            "    name = 'attr_p'\n"
            "    description = 'Attribute plugin'\n"
            "    def on_turn_completed(self, ctx, settings):\n"
            "        return {'status': 'stub_ok', 'plugin_name': 'attr_p', 'invoked': True, 'prompt_used': ctx.user_message}\n"
            "PLUGIN = _P()\n",
            encoding="utf-8",
        )
        # 2. register(registry) function pattern
        (p_dir / "reg_plugin.py").write_text(
            "class _R:\n"
            "    name = 'reg_p'\n"
            "    description = 'Registered plugin'\n"
            "    def on_turn_completed(self, ctx, settings):\n"
            "        return {'status': 'generated', 'plugin_name': 'reg_p', 'invoked': True, 'image_url': '/img/1.png'}\n"
            "def register(reg):\n"
            "    reg.register(_R())\n",
            encoding="utf-8",
        )
        # 3. Broken syntax plugin file (must not crash discovery)
        (p_dir / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        # 4. Ignored underscored and non-py files
        (p_dir / "_helper.py").write_text("PLUGIN = None\n", encoding="utf-8")
        (p_dir / "readme.txt").write_text("ignore me\n", encoding="utf-8")
        # 5. Crashing runtime plugin
        (p_dir / "crash_plugin.py").write_text(
            "class _C:\n"
            "    name = 'crash_p'\n"
            "    description = 'Crashing plugin'\n"
            "    def on_turn_completed(self, ctx, settings):\n"
            "        raise RuntimeError('Simulated CUDA OOM')\n"
            "PLUGIN = _C()\n",
            encoding="utf-8",
        )

        reg = T2IRegistry(register_builtins=False)
        reg.register(NoopT2IPlugin())
        discovered = reg.discover_directory(p_dir)
        self.assertIn("attr_p", discovered)
        self.assertIn("reg_p", discovered)
        self.assertIn("crash_p", discovered)
        self.assertTrue(reg.discovery_errors)

        disp = T2IHookDispatcher(registry=reg)
        ctx = self._sample_context()
        r1 = disp.dispatch(ctx, {"enabled": True, "active_plugin": "attr_p", "plugins_dir": str(p_dir)})
        self.assertEqual(r1.status, "stub_ok")
        self.assertEqual(r1.prompt_used, "一起看夜景吧")

        r2 = disp.dispatch(ctx, {"enabled": True, "active_plugin": "reg_p", "plugins_dir": str(p_dir)})
        self.assertEqual(r2.status, "generated")
        self.assertEqual(r2.image_url, "/img/1.png")

        r3 = disp.dispatch(ctx, {"enabled": True, "active_plugin": "crash_p", "plugins_dir": str(p_dir)})
        self.assertEqual(r3.status, "error")
        self.assertIn("Simulated CUDA OOM", r3.message)

    def test_chat_service_integration_with_t2i_dispatcher(self) -> None:
        cfg_path = self.tmp_dir / "config.json"
        store = ConfigStore(cfg_path)
        store.load()
        dispatcher = T2IHookDispatcher(config_getter=store.load)
        chat_svc = ChatService(_DummyEngine(), store, t2i_dispatcher=dispatcher)

        # Turn 1: T2I disabled by default
        res1 = chat_svc.chat("你好", session_id="s-int")
        self.assertEqual(res1["t2i_hook"]["status"], "disabled")
        self.assertIs(res1["t2i_hook"]["invoked"], False)

        # Enable example_stub dynamically
        store.update({"t2i": {"enabled": True, "active_plugin": "example_stub", "plugin_settings": {"style": "oil"}}})
        res2 = chat_svc.chat("看星空", session_id="s-int")
        self.assertEqual(res2["t2i_hook"]["status"], "stub_ok")
        self.assertIs(res2["t2i_hook"]["invoked"], True)
        self.assertEqual(res2["t2i_hook"]["plugin_name"], "example_stub")
        self.assertIn("oil", res2["t2i_hook"]["prompt_used"])


if __name__ == "__main__":
    unittest.main()
