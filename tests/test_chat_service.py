"""Unit and integration tests for backend.chat_service."""

from __future__ import annotations

import copy
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from backend.chat_service import ChatError, ChatService, SceneContext
from backend.config_store import ConfigStore, default_config
from backend.llama_engine import EngineError, EngineStatus, create_engine
from backend.skill_loader import SkillLoadError, SkillLoader

FIREFLY_DIR = Path(r"C:\Users\nzp18\Desktop\角色skill\流萤\firefly-skill")


class _FakeEngine:
    """In-memory fake LlamaEngine supporting configurable responses, faults, and call capture."""

    def __init__(self, llama_cfg: dict[str, Any] | None = None) -> None:
        self.llama_cfg = dict(llama_cfg or {})
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self.raw_responses: list[Any] = []
        self.error_to_raise: Exception | None = None
        self.delay_sec: float = 0.0
        self.shutdown_count: int = 0
        self.ensure_ready_count: int = 0
        self._lock = threading.Lock()
        self._status = EngineStatus(
            state="stopped",
            mode=str(self.llama_cfg.get("engine_mode", "external_server")),
            endpoint=str(self.llama_cfg.get("server_url", "http://127.0.0.1:8080")),
        )

    def status(self) -> EngineStatus:
        with self._lock:
            return EngineStatus(
                state=self._status.state,
                mode=self._status.mode,
                endpoint=self._status.endpoint,
                pid=self._status.pid,
                error_code=self._status.error_code,
                detail=self._status.detail,
            )

    def ensure_ready(self) -> None:
        with self._lock:
            self.ensure_ready_count += 1
            self._status.state = "ready"

    def shutdown(self) -> None:
        with self._lock:
            self.shutdown_count += 1
            self._status.state = "stopped"

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        with self._lock:
            self.calls.append((copy.deepcopy(messages), dict(params or {})))
            if self.error_to_raise is not None:
                raise self.error_to_raise
            if self.raw_responses:
                return self.raw_responses.pop(0)
            self._status.state = "ready"
            last_user = messages[-1]["content"]
            return {
                "reply": f"reply:{last_user}",
                "usage": {"prompt_tokens": len(messages) * 5, "completion_tokens": 3},
            }


class _StubT2IDispatcher:
    """Stub T2I dispatcher recording SceneContext and t2i_config invocations."""

    def __init__(self, *, should_raise: bool = False) -> None:
        self.should_raise = should_raise
        self.dispatches: list[tuple[SceneContext, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def dispatch(self, context: SceneContext, t2i_config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.dispatches.append((context, copy.deepcopy(t2i_config)))
        if self.should_raise:
            raise RuntimeError("boom in T2I plugin")
        if not t2i_config.get("enabled", False):
            return {
                "status": "disabled",
                "plugin_name": t2i_config.get("active_plugin", "noop"),
                "invoked": False,
                "prompt_used": "",
                "image_url": None,
                "image_path": None,
                "elapsed_ms": 0.1,
                "message": "T2I disabled.",
                "extra": {},
            }
        return {
            "status": "stub_ok",
            "plugin_name": t2i_config.get("active_plugin", "example_stub"),
            "invoked": True,
            "prompt_used": context.scene_prompt_hint,
            "image_url": None,
            "image_path": None,
            "elapsed_ms": 1.2,
            "message": "Stub dispatched.",
            "extra": {"turn_index": context.turn_index},
        }


class _OneArgT2IDispatcher:
    """Stub dispatcher accepting only (context) to verify signature detection."""

    def __init__(self) -> None:
        self.contexts: list[SceneContext] = []

    def dispatch(self, context: SceneContext) -> dict[str, Any]:
        self.contexts.append(context)
        return {
            "status": "stub_ok",
            "plugin_name": "one_arg",
            "invoked": True,
            "prompt_used": context.scene_prompt_hint,
        }


class _MockConfigStore:
    """In-memory config store allowing zero-boundary testing beyond ConfigStore's >=1 validation."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = copy.deepcopy(data)

    def load(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        for section, vals in patch.items():
            if isinstance(vals, dict) and isinstance(self._data.get(section), dict):
                self._data[section].update(copy.deepcopy(vals))
            else:
                self._data[section] = copy.deepcopy(vals)
        return self.load()


class ChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.config_path = self.tmp_dir / "config.json"
        self.store = ConfigStore(self.config_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _create_skill_dir(self, name: str, prompt_text: str) -> Path:
        skill_dir = self.tmp_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill\n---\n{prompt_text}\n",
            encoding="utf-8",
        )
        (skill_dir / "README.md").write_text("# Meta Readme\nMeta notes.\n", encoding="utf-8")
        (skill_dir / "LICENSE").write_text("MIT", encoding="utf-8")
        return skill_dir

    # T01
    def test_chat_without_skill_single_and_multi_turn(self) -> None:
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        self.assertIsNone(svc.current_skill())
        self.assertEqual(svc.current_skill_dict(), {"loaded": False})

        r1 = svc.chat("Hello there", session_id="s1")
        self.assertEqual(r1["session_id"], "s1")
        self.assertEqual(r1["turn_index"], 1)
        self.assertEqual(r1["reply"], "reply:Hello there")
        self.assertEqual(r1["messages_count"], 2)
        self.assertEqual(engine.calls[-1][0], [{"role": "user", "content": "Hello there"}])

        r2 = svc.send_turn("How are you?", session_id="s1")
        self.assertEqual(r2["turn_index"], 2)
        self.assertEqual(r2["messages_count"], 4)
        self.assertEqual(
            engine.calls[-1][0],
            [
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "reply:Hello there"},
                {"role": "user", "content": "How are you?"},
            ],
        )

    # T02
    def test_load_skill_injects_system_prompt_and_uses_config_defaults(self) -> None:
        skill_dir = self._create_skill_dir("Aria", "Speak gently like Aria.")
        self.store.update({"skill": {"exclude_meta_docs": True, "max_prompt_chars": 5000}})
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        loaded = svc.load_skill(skill_dir)
        self.assertEqual(loaded.character_name, "Aria")
        self.assertNotIn("Meta notes.", loaded.system_prompt)
        self.assertEqual(svc.current_skill().character_name, "Aria")
        self.assertTrue(svc.current_skill_dict()["loaded"])
        self.assertEqual(svc.current_skill_dict()["character_name"], "Aria")
        self.assertEqual(self.store.load()["skill"]["last_path"], str(skill_dir.resolve()))

        svc.chat("Hi Aria", session_id="s-aria")
        sent_messages, sent_params = engine.calls[-1]
        self.assertEqual(sent_messages[0]["role"], "system")
        self.assertIn("Speak gently like Aria.", sent_messages[0]["content"])
        self.assertEqual(sent_messages[1], {"role": "user", "content": "Hi Aria"})
        self.assertEqual(sent_params["temperature"], 0.8)
        self.assertEqual(sent_params["top_p"], 0.9)
        self.assertEqual(sent_params["max_tokens"], 512)

        # restore_last_skill works on fresh service instance
        svc2 = ChatService(engine, self.store, SkillLoader)
        restored = svc2.restore_last_skill()
        self.assertIsNotNone(restored)
        self.assertEqual(restored.character_name, "Aria")

    # T03
    def test_load_skill_failure_preserves_previous_skill(self) -> None:
        skill_dir = self._create_skill_dir("Luna", "Luna system prompt.")
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)
        svc.load_skill(skill_dir)

        with self.assertRaises(SkillLoadError):
            svc.load_skill(self.tmp_dir / "nonexistent_skill")

        self.assertEqual(svc.current_skill().character_name, "Luna")
        self.assertEqual(self.store.load()["skill"]["last_path"], str(skill_dir.resolve()))

        # Stale last_path in restore_last_skill does not crash
        self.store.update({"skill": {"last_path": str(self.tmp_dir / "deleted_skill")}})
        svc2 = ChatService(engine, self.store, SkillLoader)
        self.assertIsNone(svc2.restore_last_skill())

    # T04
    def test_skill_hot_swap_updates_active_session_system_prompt(self) -> None:
        s_a = self._create_skill_dir("CharA", "Persona A")
        s_b = self._create_skill_dir("CharB", "Persona B")
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        svc.load_skill(s_a)
        svc.chat("Turn 1", session_id="swap")
        self.assertIn("Persona A", engine.calls[-1][0][0]["content"])

        svc.load_skill(s_b)
        svc.chat("Turn 2", session_id="swap")
        sent = engine.calls[-1][0]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("Persona B", sent[0]["content"])
        self.assertEqual(sent[1]["content"], "Turn 1")
        self.assertEqual(sent[2]["content"], "reply:Turn 1")
        self.assertEqual(sent[3]["content"], "Turn 2")

    # T05
    def test_max_history_turns_sliding_window_and_zero_boundary(self) -> None:
        skill_dir = self._create_skill_dir("Pinned", "Pinned system.")
        self.store.update({"llama": {"max_history_turns": 2}})
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)
        svc.load_skill(skill_dir)

        for idx in range(1, 5):
            res = svc.chat(f"msg-{idx}", session_id="win")
            self.assertEqual(res["turn_index"], idx)

        sent = engine.calls[-1][0]
        # system + 2 prior turns (4 msgs) + current user (1 msg) = 6
        self.assertEqual(len(sent), 6)
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual(
            [(m["role"], m["content"]) for m in sent[1:]],
            [
                ("user", "msg-2"),
                ("assistant", "reply:msg-2"),
                ("user", "msg-3"),
                ("assistant", "reply:msg-3"),
                ("user", "msg-4"),
            ],
        )

        # Zero boundary via mock config store
        zero_cfg = default_config()
        zero_cfg["llama"]["max_history_turns"] = 0
        mock_store = _MockConfigStore(zero_cfg)
        engine2 = _FakeEngine()
        svc_zero = ChatService(engine2, mock_store, SkillLoader)
        svc_zero.load_skill(skill_dir)
        svc_zero.chat("first", session_id="z")
        svc_zero.chat("second", session_id="z")
        sent_zero = engine2.calls[-1][0]
        self.assertEqual(len(sent_zero), 2)
        self.assertEqual(sent_zero[0]["role"], "system")
        self.assertEqual(sent_zero[1], {"role": "user", "content": "second"})

    # T06
    def test_session_isolation_auto_id_and_selective_reset(self) -> None:
        skill_dir = self._create_skill_dir("KeepChar", "Keep this character.")
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)
        svc.load_skill(skill_dir)

        auto_r1 = svc.chat("auto-1")
        sid = auto_r1["session_id"]
        self.assertTrue(isinstance(sid, str) and len(sid) > 0)
        auto_r2 = svc.chat("auto-2", session_id=sid)
        self.assertEqual(auto_r2["turn_index"], 2)

        svc.chat("other-1", session_id="s-other")
        reset_res = svc.reset_session(sid)
        self.assertEqual(reset_res["status"], "ok")
        self.assertEqual(reset_res["session_id"], sid)
        self.assertTrue(reset_res["skill_loaded"])

        self.assertEqual(svc.get_session_history(sid), [])
        self.assertEqual(len(svc.get_session_history("s-other")), 2)
        self.assertEqual(svc.current_skill().character_name, "KeepChar")

        next_after_reset = svc.chat("after-reset", session_id=sid)
        self.assertEqual(next_after_reset["turn_index"], 1)

        # Resetting nonexistent session is idempotent
        unknown_reset = svc.reset_session("never-existed")
        self.assertEqual(unknown_reset["status"], "ok")

        # Global reset clears all sessions while keeping skill
        all_reset = svc.reset_session()
        self.assertIsNone(all_reset["session_id"])
        self.assertEqual(svc.get_session_history("s-other"), [])
        self.assertEqual(svc.current_skill().character_name, "KeepChar")

    # T07
    def test_validation_empty_whitespace_and_invalid_types(self) -> None:
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        for bad_msg in ("", "   ", "\n\t  ", None, 123):
            with self.subTest(bad_msg=bad_msg):
                with self.assertRaises(ChatError) as ctx:
                    svc.chat(bad_msg, session_id="v1")  # type: ignore[arg-type]
                self.assertEqual(ctx.exception.code, "CHAT_INVALID_MESSAGE")
                self.assertEqual(ctx.exception.http_status, 400)

        with self.assertRaises(ChatError) as ctx:
            svc.chat("ok", session_id=12345)  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.code, "CHAT_INVALID_SESSION_ID")
        self.assertEqual(ctx.exception.http_status, 400)

        self.assertEqual(len(engine.calls), 0)
        self.assertEqual(svc.get_session_history("v1"), [])

        # Valid padded/CJK/emoji message preserved verbatim
        verbatim_msg = "  你好 流萤 🌟 \n"
        svc.chat(verbatim_msg, session_id="v1")
        self.assertEqual(engine.calls[-1][0][-1]["content"], verbatim_msg)
        self.assertEqual(svc.get_session_history("v1")[0]["content"], verbatim_msg)

    # T08
    def test_failure_atomicity_on_engine_error_and_malformed_reply(self) -> None:
        engine = _FakeEngine()
        t2i = _StubT2IDispatcher()
        svc = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=t2i)

        svc.chat("good turn", session_id="atom")
        self.assertEqual(len(t2i.dispatches), 1)

        # EngineError leaves history untouched and does not invoke T2I
        engine.error_to_raise = EngineError("ENGINE_UPSTREAM_ERROR", "upstream down", http_status=502)
        with self.assertRaises(EngineError):
            svc.chat("failing turn", session_id="atom")
        self.assertEqual(len(svc.get_session_history("atom")), 2)
        self.assertEqual(len(t2i.dispatches), 1)

        # Malformed reply (empty choices) also rolls back cleanly
        engine.error_to_raise = None
        engine.raw_responses.append({"choices": []})
        with self.assertRaises(ChatError) as ctx:
            svc.chat("malformed turn", session_id="atom")
        self.assertEqual(ctx.exception.http_status, 502)
        self.assertEqual(len(svc.get_session_history("atom")), 2)
        self.assertEqual(len(t2i.dispatches), 1)

        # Subsequent valid turn gets turn_index == 2 and strict alternation
        r2 = svc.chat("recovered turn", session_id="atom")
        self.assertEqual(r2["turn_index"], 2)
        history = svc.get_session_history("atom")
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "user", "assistant"])

    # T09
    def test_engine_factory_reload_and_lifecycle_helpers(self) -> None:
        created_engines: list[_FakeEngine] = []

        def factory(llama_cfg: dict[str, Any]) -> _FakeEngine:
            eng = _FakeEngine(llama_cfg)
            created_engines.append(eng)
            return eng

        svc = ChatService(factory, self.store, SkillLoader)
        self.assertEqual(svc.engine_status().state, "stopped")
        self.assertEqual(len(created_engines), 1)

        st = svc.start_engine()
        self.assertEqual(st.state, "ready")
        self.assertEqual(created_engines[0].ensure_ready_count, 1)

        # Changing sampling param alone does NOT recreate engine
        svc.update_config({"llama": {"temperature": 0.42}})
        self.assertEqual(len(created_engines), 1)
        self.assertEqual(created_engines[0].shutdown_count, 0)
        svc.chat("check temp", session_id="cfg")
        self.assertEqual(created_engines[0].calls[-1][1]["temperature"], 0.42)

        # Changing engine-binding param shuts down old engine and instantiates new one
        svc.update_config({"llama": {"engine_mode": "external_server", "server_url": "http://127.0.0.1:9090"}})
        self.assertEqual(created_engines[0].shutdown_count, 1)
        self.assertEqual(len(created_engines), 2)
        self.assertEqual(created_engines[1].llama_cfg["server_url"], "http://127.0.0.1:9090")

        stopped = svc.stop_engine()
        self.assertEqual(stopped.state, "stopped")
        self.assertEqual(created_engines[1].shutdown_count, 1)

        svc.shutdown()
        self.assertEqual(created_engines[1].shutdown_count, 2)

    # T10
    def test_t2i_dispatcher_none_stub_and_exception_isolation(self) -> None:
        engine = _FakeEngine()

        # 1. t2i_dispatcher=None
        svc_none = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=None)
        res_none = svc_none.chat("hi", session_id="t-none")
        self.assertEqual(res_none["t2i_hook"]["status"], "disabled")
        self.assertFalse(res_none["t2i_hook"]["invoked"])

        # 2. Stub dispatcher with dynamic enabled toggle & context_window_turns capping
        self.store.update({"t2i": {"enabled": True, "active_plugin": "example_stub", "context_window_turns": 1}})
        skill_dir = self._create_skill_dir("Camellia", "Camellia prompt.")
        stub = _StubT2IDispatcher(should_raise=False)
        svc_stub = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=stub)
        svc_stub.load_skill(skill_dir)

        svc_stub.chat("turn1", session_id="t-stub")
        res2 = svc_stub.chat("turn2", session_id="t-stub")
        self.assertEqual(res2["t2i_hook"]["status"], "stub_ok")
        self.assertTrue(res2["t2i_hook"]["invoked"])

        last_ctx, last_cfg = stub.dispatches[-1]
        self.assertIsInstance(last_ctx, SceneContext)
        self.assertEqual(last_ctx.session_id, "t-stub")
        self.assertEqual(last_ctx.turn_index, 2)
        self.assertEqual(last_ctx.character_name, "Camellia")
        self.assertEqual(last_ctx.skill_source, str(skill_dir.resolve()))
        self.assertEqual(last_ctx.user_message, "turn2")
        self.assertEqual(last_ctx.assistant_reply, "reply:turn2")
        self.assertEqual(len(last_ctx.recent_messages), 2)
        self.assertTrue(last_ctx.timestamp_utc.endswith("Z"))
        self.assertIn("Camellia", last_ctx.scene_prompt_hint)
        self.assertTrue(last_cfg["enabled"])

        # Dynamic toggle off -> disabled
        svc_stub.update_config({"t2i": {"enabled": False}})
        res3 = svc_stub.chat("turn3", session_id="t-stub")
        self.assertEqual(res3["t2i_hook"]["status"], "disabled")
        self.assertFalse(res3["t2i_hook"]["invoked"])

        # 1-arg dispatcher support
        one_arg = _OneArgT2IDispatcher()
        svc_one = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=one_arg)
        res_one = svc_one.chat("one-arg", session_id="t-one")
        self.assertEqual(res_one["t2i_hook"]["plugin_name"], "one_arg")
        self.assertEqual(len(one_arg.contexts), 1)

        # 3. Crashing dispatcher is isolated; turn commits and reply returns
        crashing = _StubT2IDispatcher(should_raise=True)
        svc_crash = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=crashing)
        res_crash = svc_crash.chat("stay alive", session_id="t-crash")
        self.assertEqual(res_crash["reply"], "reply:stay alive")
        self.assertEqual(res_crash["turn_index"], 1)
        self.assertEqual(res_crash["t2i_hook"]["status"], "error")
        self.assertIn("boom in T2I plugin", res_crash["t2i_hook"]["message"])
        self.assertEqual(len(svc_crash.get_session_history("t-crash")), 2)

    # T11
    def test_concurrent_chats_across_and_within_sessions(self) -> None:
        engine = _FakeEngine()
        engine.delay_sec = 0.02
        svc = ChatService(engine, self.store, SkillLoader)

        def worker(sid: str, idx: int) -> dict[str, Any]:
            return svc.chat(f"{sid}-msg-{idx}", session_id=sid)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for s_idx in range(4):
                sid = f"conc-{s_idx}"
                for m_idx in range(3):
                    futures.append(pool.submit(worker, sid, m_idx))
            results = [f.result(timeout=10.0) for f in futures]

        self.assertEqual(len(results), 12)
        for s_idx in range(4):
            sid = f"conc-{s_idx}"
            hist = svc.get_session_history(sid)
            self.assertEqual(len(hist), 6)
            self.assertEqual([m["role"] for m in hist], ["user", "assistant"] * 3)

    # T12
    @unittest.skipUnless(FIREFLY_DIR.exists(), "Real firefly-skill fixture directory not present")
    def test_integration_real_firefly_skill_with_config_store(self) -> None:
        self.store.update({"skill": {"exclude_meta_docs": True}, "t2i": {"enabled": True}})
        engine = _FakeEngine()
        stub = _StubT2IDispatcher()
        svc = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=stub)

        loaded = svc.load_skill(FIREFLY_DIR)
        self.assertEqual(loaded.character_name, "firefly-hsr")
        self.assertIn("LICENSE", loaded.ignored_files)
        self.assertIn("manifest.json", loaded.ignored_files)
        self.assertIn("README.md", loaded.excluded_files)
        self.assertNotIn("MIT", loaded.system_prompt)

        r1 = svc.chat("流萤，今晚去看星星吗？", session_id="ff-1")
        self.assertEqual(r1["turn_index"], 1)
        self.assertEqual(engine.calls[-1][0][0]["role"], "system")
        self.assertEqual(stub.dispatches[-1][0].character_name, "firefly-hsr")

    # T13
    def test_integration_with_create_engine_factory(self) -> None:
        svc = ChatService(create_engine, self.store, SkillLoader)
        st = svc.engine_status()
        self.assertEqual(st.state, "stopped")
        self.assertEqual(st.mode, "managed_server")

        svc.update_config({"llama": {"engine_mode": "external_server", "server_url": ""}})
        with self.assertRaises(EngineError) as ctx:
            svc.start_engine()
        self.assertEqual(ctx.exception.code, "CONFIG_INCOMPLETE")
        self.assertEqual(ctx.exception.http_status, 400)
        svc.shutdown()

    # T14
    def test_chat_invalid_params_raise_chat_invalid_argument_400(self) -> None:
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        bad_cases: list[Any] = [
            "bad",
            ["not", "a", "mapping"],
            {"temperature": "bad"},
            {"temperature": None},
            {"temperature": True},
            {"temperature": float("nan")},
            {"top_p": float("inf")},
            {"max_tokens": 0},
            {"max_tokens": -2},
            {"max_tokens": True},
            {"max_tokens": 1.5},
        ]
        for bad_params in bad_cases:
            with self.subTest(bad_params=bad_params):
                with self.assertRaises(ChatError) as ctx:
                    svc.chat("hello", session_id="bad-p", params=bad_params)  # type: ignore[arg-type]
                self.assertEqual(ctx.exception.code, "CHAT_INVALID_ARGUMENT")
                self.assertEqual(ctx.exception.http_status, 400)

        self.assertEqual(len(engine.calls), 0)
        self.assertEqual(svc.get_session_history("bad-p"), [])

    # T15
    def test_load_skill_preserves_skill_loader_strict_argument_type_errors(self) -> None:
        skill_dir = self._create_skill_dir("StrictSkill", "Strict persona.")
        engine = _FakeEngine()
        svc = ChatService(engine, self.store, SkillLoader)

        for bad_exclude in ("yes", 1, []):
            with self.subTest(bad_exclude=bad_exclude):
                with self.assertRaises(SkillLoadError) as ctx:
                    svc.load_skill(skill_dir, exclude_meta_docs=bad_exclude)  # type: ignore[arg-type]
                self.assertEqual(ctx.exception.code, "SKILL_INVALID_ARGUMENT")
                self.assertEqual(ctx.exception.http_status, 400)

        for bad_max_chars in (True, -1, "100", 1.5):
            with self.subTest(bad_max_chars=bad_max_chars):
                with self.assertRaises(SkillLoadError) as ctx:
                    svc.load_skill(skill_dir, max_prompt_chars=bad_max_chars)  # type: ignore[arg-type]
                self.assertEqual(ctx.exception.code, "SKILL_INVALID_ARGUMENT")
                self.assertEqual(ctx.exception.http_status, 400)

        self.assertIsNone(svc.current_skill())

    # T16
    def test_t2i_dispatcher_elapsed_ms_none_and_custom_positional_kwargs_signature(self) -> None:
        engine = _FakeEngine()

        class _NoneElapsedDictDispatcher:
            def dispatch(self, context: SceneContext, t2i_config: dict[str, Any]) -> dict[str, Any]:
                return {
                    "status": "stub_ok",
                    "plugin_name": "none_elapsed_dict",
                    "invoked": True,
                    "prompt_used": context.scene_prompt_hint,
                    "elapsed_ms": None,
                }

        svc1 = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=_NoneElapsedDictDispatcher())
        res1 = svc1.chat("scene 1", session_id="t16-1")
        self.assertEqual(res1["t2i_hook"]["status"], "stub_ok")
        self.assertTrue(res1["t2i_hook"]["invoked"])
        self.assertIsInstance(res1["t2i_hook"]["elapsed_ms"], float)
        self.assertGreaterEqual(res1["t2i_hook"]["elapsed_ms"], 0.0)

        class _NoneElapsedObj:
            status = "stub_ok"
            plugin_name = "none_elapsed_obj"
            invoked = True
            prompt_used = "hint"
            image_url = None
            image_path = None
            elapsed_ms = None
            message = "ok"
            extra = {}

        class _CustomPositionalKwargsDispatcher:
            def __init__(self) -> None:
                self.calls: list[tuple[SceneContext, dict[str, Any], dict[str, Any]]] = []

            def dispatch(self, context: SceneContext, cfg: dict[str, Any], **kwargs: Any) -> _NoneElapsedObj:
                self.calls.append((context, cfg, kwargs))
                return _NoneElapsedObj()

        custom_disp = _CustomPositionalKwargsDispatcher()
        svc2 = ChatService(engine, self.store, SkillLoader, t2i_dispatcher=custom_disp)
        res2 = svc2.chat("scene 2", session_id="t16-2")
        self.assertEqual(res2["t2i_hook"]["status"], "stub_ok")
        self.assertEqual(res2["t2i_hook"]["plugin_name"], "none_elapsed_obj")
        self.assertTrue(res2["t2i_hook"]["invoked"])
        self.assertIsInstance(res2["t2i_hook"]["elapsed_ms"], float)
        self.assertEqual(len(custom_disp.calls), 1)
        self.assertEqual(custom_disp.calls[0][2], {})


if __name__ == "__main__":
    unittest.main()
