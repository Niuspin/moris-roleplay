"""Tier 4 E2E tests: Realistic end-to-end user journeys T4-01..T4-06 (6 tests, zero skips)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import unittest

from e2e_tests.harness import (
    AEMEATH_SKILL_DIR,
    AppProcessHarness,
    FIREFLY_SKILL_DIR,
    FakeLlamaServer,
    REPO_ROOT,
    YAE_MIKO_SKILL_FILE,
    find_browser_executable,
    is_port_open,
    make_recording_plugin_source,
    read_jsonl,
    require_target_file,
    run_headless_edge_dom,
    write_t2i_plugin,
)


class TestTier4Scenarios(unittest.TestCase):
    def _make_temp_dir(self, prefix: str = "moris_t4_") -> Path:
        d = Path(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return d

    def _start_fake_llama(self, **kwargs) -> FakeLlamaServer:
        srv = FakeLlamaServer(**kwargs).start()
        self.addCleanup(srv.stop)
        return srv

    def test_t4_01_first_time_start_bat_onboarding_firefly_roleplay(self) -> None:
        tmp = self._make_temp_dir("moris_t4_01_")
        cfg_path = tmp / "first_run_config.json"
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"开拓者，这是第 {idx} 轮回应。")

        app = AppProcessHarness(
            use_bat_launcher=True,
            config_path=cfg_path,
            write_config=False,
        ).start()
        port = app.port
        try:
            self.assertTrue(cfg_path.is_file())
            self.assertEqual(app.get_text("/").status, 200)
            self.assertEqual(app.get_text("/static/style.css").status, 200)
            self.assertEqual(app.get_text("/static/app.js").status, 200)

            put = app.put_json(
                "/api/config",
                {"llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}},
            )
            self.assertEqual(put.status, 200)

            load = app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
            self.assertEqual(load.status, 200)
            self.assertEqual(len(load.data["loaded_files"]), 11)
            self.assertIn("LICENSE", load.data["ignored_files"])
            self.assertIn("manifest.json", load.data["ignored_files"])
            self.assertNotIn("Copyright (c) 2026 Heart Ease", load.data["system_prompt"])
            self.assertNotIn("evidence_summary", load.data["system_prompt"])

            for turn in range(1, 4):
                chat = app.post_json(
                    "/api/chat",
                    {"session_id": "onboarding", "message": f"流萤，第{turn}句问候"},
                )
                self.assertEqual(chat.status, 200)
                self.assertEqual(chat.data["turn_index"], turn)
                self.assertEqual(chat.data["t2i_hook"]["status"], "disabled")
                self.assertIs(chat.data["t2i_hook"]["invoked"], False)
        finally:
            app.stop()

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and is_port_open("127.0.0.1", port):
            time.sleep(0.1)
        self.assertFalse(is_port_open("127.0.0.1", port))

    def test_t4_02_long_multi_turn_sliding_window_and_t2i_scene_recorder(self) -> None:
        tmp = self._make_temp_dir("moris_t4_02_")
        log_path = tmp / "scene_recorder.jsonl"
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"Firefly-Turn-{idx}")

        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "max_history_turns": 2,
                },
                "t2i": {
                    "enabled": True,
                    "active_plugin": "scene_recorder",
                    "context_window_turns": 2,
                },
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "scene_recorder.py",
            make_recording_plugin_source("scene_recorder", log_path),
        )
        app.start()
        self.addCleanup(app.stop)

        self.assertEqual(app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)}).status, 200)

        for idx in range(1, 7):
            r = app.post_json("/api/chat", {"session_id": "long-rp", "message": f"User-Turn-{idx}"})
            self.assertEqual(r.status, 200)
            self.assertEqual(r.data["t2i_hook"]["status"], "stub_ok")

        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("绝对不可违背的最高规则", msgs[0]["content"])
        non_sys = [m["content"] for m in msgs[1:]]
        for evicted in ("User-Turn-1", "User-Turn-2", "User-Turn-3"):
            self.assertNotIn(evicted, non_sys)
        self.assertEqual(
            non_sys,
            [
                "User-Turn-4",
                "Firefly-Turn-4",
                "User-Turn-5",
                "Firefly-Turn-5",
                "User-Turn-6",
            ],
        )

        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 6)
        last_ctx = entries[-1]
        self.assertEqual(last_ctx["turn_index"], 6)
        self.assertEqual(last_ctx["character_name"], "firefly-hsr")
        self.assertLessEqual(len(last_ctx["recent_messages"]), 4)

    def test_t4_03_character_switching_and_session_reset_journey(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"char-reply-{idx}")
        app = AppProcessHarness(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        ).start()
        self.addCleanup(app.stop)

        # 1. Yae Miko single-file skill
        l1 = app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)})
        self.assertEqual(l1.status, 200)
        self.assertEqual(l1.data["character_name"], "yae-miko-perspective")
        self.assertEqual(app.post_json("/api/chat", {"session_id": "journey", "message": "八重你好"}).status, 200)
        self.assertEqual(app.post_json("/api/chat", {"session_id": "journey", "message": "聊聊轻小说"}).status, 200)

        # 2. Reset session and switch to Firefly directory skill
        self.assertEqual(app.post_json("/api/sessions/reset", {"session_id": "journey"}).status, 200)
        l2 = app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
        self.assertEqual(l2.status, 200)
        c3 = app.post_json("/api/chat", {"session_id": "journey", "message": "流萤你好"})
        self.assertEqual(c3.status, 200)
        self.assertEqual(c3.data["turn_index"], 1)
        m3 = fake_llama.last_request["body"]["messages"]
        self.assertIn("绝对不可违背的最高规则", m3[0]["content"])
        self.assertNotIn("八重你好", json.dumps(m3, ensure_ascii=False))

        # 3. Switch to Aemeath directory skill (verifying top-level-only scan)
        l3 = app.post_json("/api/skill/load", {"path": str(AEMEATH_SKILL_DIR)})
        self.assertEqual(l3.status, 200)
        self.assertEqual(l3.data["character_name"], "aemeath")
        self.assertNotIn("# 爱弥斯行为蒸馏链", l3.data["system_prompt"])
        c4 = app.post_json("/api/chat", {"session_id": "journey", "message": "爱弥斯你好"})
        self.assertEqual(c4.status, 200)
        self.assertEqual(fake_llama.last_request["body"]["messages"][0]["content"], l3.data["system_prompt"])

    def test_t4_04_misconfiguration_and_fault_recovery_journey(self) -> None:
        tmp = self._make_temp_dir("moris_t4_04_")
        app = AppProcessHarness(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(tmp / "missing_llama.exe"),
                    "model_path": str(tmp / "missing_model.gguf"),
                }
            }
        ).start()
        self.addCleanup(app.stop)

        bad_start = app.post_json("/api/engine/start", {})
        self.assertEqual(bad_start.status, 400)
        self.assertIn("error", bad_start.data)
        bad_chat = app.post_json("/api/chat", {"session_id": "rec", "message": "hi"})
        self.assertEqual(bad_chat.status, 400)
        self.assertEqual(app.get_json("/api/health").status, 200)

        fake_llama = self._start_fake_llama(health_sequence=[503, 503, 200], default_reply="恢复完成")
        app.put_json(
            "/api/config",
            {
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "startup_timeout_sec": 5,
                }
            },
        )
        # First /api/engine/start hits 503 loading; after sequence exhausts, recovers to 200 ready
        s1 = app.post_json("/api/engine/start", {})
        if s1.status == 503:
            s2 = app.post_json("/api/engine/start", {})
            if s2.status == 503:
                s3 = app.post_json("/api/engine/start", {})
                self.assertEqual(s3.status, 200)
                self.assertEqual(s3.data["state"], "ready")
            else:
                self.assertEqual(s2.status, 200)
        else:
            self.assertEqual(s1.status, 200)

        self.assertEqual(app.post_json("/api/chat", {"session_id": "rec", "message": "Turn1"}).status, 200)
        fake_llama.enqueue_response(mode="http_500")
        self.assertEqual(app.post_json("/api/chat", {"session_id": "rec", "message": "Fail500"}).status, 502)
        fake_llama.enqueue_response(mode="malformed_json")
        self.assertEqual(app.post_json("/api/chat", {"session_id": "rec", "message": "FailJSON"}).status, 502)

        ok = app.post_json("/api/chat", {"session_id": "rec", "message": "Turn2-OK"})
        self.assertEqual(ok.status, 200)
        self.assertEqual(ok.data["turn_index"], 2)
        dump = json.dumps(fake_llama.last_request["body"]["messages"], ensure_ascii=False)
        self.assertNotIn("Fail500", dump)
        self.assertNotIn("FailJSON", dump)

    def test_t4_05_full_restart_persistence_and_dropin_plugin_lifecycle(self) -> None:
        tmp = self._make_temp_dir("moris_t4_05_")
        cfg_path = tmp / "lifecycle_cfg.json"
        plugins_dir = tmp / "plugins" / "t2i"
        log_path = tmp / "lifecycle_t2i.jsonl"
        write_t2i_plugin(
            plugins_dir,
            "lifecycle_rec.py",
            make_recording_plugin_source("lifecycle_rec", log_path),
        )
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"session-reply-{idx}")

        # Process lifetime 1
        app1 = AppProcessHarness(config_path=cfg_path).start()
        try:
            self.assertEqual(
                app1.put_json(
                    "/api/config",
                    {
                        "llama": {
                            "engine_mode": "external_server",
                            "server_url": fake_llama.base_url,
                            "temperature": 0.42,
                            "max_tokens": 300,
                        },
                        "t2i": {
                            "enabled": True,
                            "active_plugin": "lifecycle_rec",
                            "plugins_dir": str(plugins_dir),
                            "plugin_settings": {"mode": "persist"},
                        },
                    },
                ).status,
                200,
            )
            self.assertEqual(app1.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)}).status, 200)
            c1 = app1.post_json("/api/chat", {"session_id": "p1", "message": "生命周期第一轮"})
            self.assertEqual(c1.status, 200)
            self.assertEqual(c1.data["t2i_hook"]["plugin_name"], "lifecycle_rec")
        finally:
            app1.stop()

        # Process lifetime 2 (same config_path, no PUT /api/config)
        app2 = AppProcessHarness(config_path=cfg_path, write_config=False).start()
        self.addCleanup(app2.stop)
        cfg2 = app2.get_json("/api/config").data
        self.assertAlmostEqual(cfg2["llama"]["temperature"], 0.42)
        self.assertEqual(cfg2["llama"]["max_tokens"], 300)
        self.assertEqual(cfg2["skill"]["last_path"], str(FIREFLY_SKILL_DIR))
        ext2 = app2.get_json("/api/extensions/t2i").data
        self.assertIs(ext2["enabled"], True)
        self.assertEqual(ext2["active_plugin"], "lifecycle_rec")

        # Ensure skill is active (either auto-reloaded from last_path or explicitly reloaded if needed)
        curr_skill = app2.get_json("/api/skill/current").data
        if not curr_skill.get("loaded", True) or not curr_skill.get("character_name"):
            app2.post_json("/api/skill/load", {"path": cfg2["skill"]["last_path"]})

        c2 = app2.post_json("/api/chat", {"session_id": "p2", "message": "重启后第二轮"})
        self.assertEqual(c2.status, 200)
        self.assertEqual(c2.data["t2i_hook"]["status"], "stub_ok")
        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[1]["plugin_settings"], {"mode": "persist"})

        # Check docs remain consistent
        require_target_file(REPO_ROOT / "README.md")
        require_target_file(REPO_ROOT / "docs" / "EXTENSIONS.md")

    def test_t4_06_web_ui_static_offline_api_sequence_and_headless_edge_dom(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="Web UI 全流程回复")
        app = AppProcessHarness(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        ).start()
        self.addCleanup(app.stop)

        html = app.get_text("/")
        css = app.get_text("/static/style.css")
        js = app.get_text("/static/app.js")
        self.assertEqual(html.status, 200)
        self.assertEqual(css.status, 200)
        self.assertEqual(js.status, 200)
        self.assertIsNone(re.search(r"""(?:src|href)\s*=\s*["']https?://""", html.text, re.I))

        # Full frontend API sequence
        self.assertEqual(app.get_json("/api/health").status, 200)
        self.assertEqual(app.get_json("/api/config").status, 200)
        self.assertEqual(app.get_json("/api/extensions/t2i").status, 200)
        self.assertEqual(app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)}).status, 200)
        chat = app.post_json("/api/chat", {"session_id": "webui", "message": "前端联调"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["reply"], "Web UI 全流程回复")
        self.assertEqual(app.post_json("/api/sessions/reset", {"session_id": "webui"}).status, 200)

        # Headless Edge DOM verification (strict, zero skips)
        exe = find_browser_executable()
        self.assertIsNotNone(exe, "msedge.exe or chrome.exe must exist for headless DOM verification")
        rc, dom, stderr, _ = run_headless_edge_dom(app.base_url)
        self.assertEqual(rc, 0, f"Headless Edge DOM dump failed: {stderr}")
        dom_lower = dom.lower()
        self.assertIn("<html", dom_lower)
        self.assertTrue("textarea" in dom_lower or "<input" in dom_lower)
        self.assertIn("button", dom_lower)
        self.assertTrue("skill" in dom_lower or "角色" in dom)
        self.assertTrue("t2i" in dom_lower or "文生图" in dom)


if __name__ == "__main__":
    unittest.main()
