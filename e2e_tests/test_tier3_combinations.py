"""Tier 3 E2E tests: Pairwise cross-feature combinations across F1-F8 (12 tests, zero skips)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import unittest

from e2e_tests.harness import (
    AppProcessHarness,
    FIREFLY_SKILL_DIR,
    FakeLlamaServer,
    REPO_ROOT,
    find_free_port,
    is_port_open,
    make_recording_plugin_source,
    make_single_md_skill,
    read_jsonl,
    require_target_file,
    write_t2i_plugin,
)


class TestTier3Combinations(unittest.TestCase):
    def _make_temp_dir(self, prefix: str = "moris_t3_") -> Path:
        d = Path(tempfile.mkdtemp(prefix=prefix))
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return d

    def _start_fake_llama(self, **kwargs) -> FakeLlamaServer:
        srv = FakeLlamaServer(**kwargs).start()
        self.addCleanup(srv.stop)
        return srv

    def _start_app(self, **kwargs) -> AppProcessHarness:
        app = AppProcessHarness(**kwargs).start()
        self.addCleanup(app.stop)
        return app

    def test_t3_01_f1_f2_config_skill_budget_and_exclude_meta_persistence(self) -> None:
        tmp = self._make_temp_dir("moris_t3_01_")
        cfg_path = tmp / "cfg.json"
        app = self._start_app(config_path=cfg_path)

        put = app.put_json(
            "/api/config",
            {"skill": {"exclude_meta_docs": True, "max_prompt_chars": 4000}},
        )
        self.assertEqual(put.status, 200)

        load = app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
        self.assertEqual(load.status, 200)
        data = load.data
        self.assertEqual(
            sorted(data["excluded_files"]),
            sorted(["README.md", "quality-report.md", "test-report.md"]),
        )
        for meta in ("README.md", "quality-report.md", "test-report.md"):
            self.assertNotIn(meta, data["loaded_files"])
        self.assertLessEqual(data["total_chars"], 4000)
        self.assertTrue(data["warnings"])

        disk_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_cfg["skill"]["last_path"], str(FIREFLY_SKILL_DIR))

    def test_t3_02_f1_f3_sampling_params_applied_and_persist_across_restart(self) -> None:
        tmp = self._make_temp_dir("moris_t3_02_")
        cfg_path = tmp / "cfg.json"
        fake_llama = self._start_fake_llama(default_reply="采样参数测试")

        app1 = AppProcessHarness(
            config_path=cfg_path,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            },
        ).start()
        try:
            put = app1.put_json(
                "/api/config",
                {"llama": {"temperature": 0.35, "top_p": 0.75, "max_tokens": 128}},
            )
            self.assertEqual(put.status, 200)
            c1 = app1.post_json("/api/chat", {"session_id": "t3-02", "message": "first"})
            self.assertEqual(c1.status, 200)
            b1 = fake_llama.last_request["body"]
            self.assertAlmostEqual(b1["temperature"], 0.35)
            self.assertAlmostEqual(b1["top_p"], 0.75)
            self.assertEqual(b1["max_tokens"], 128)
        finally:
            app1.stop()

        app2 = self._start_app(config_path=cfg_path, write_config=False)
        c2 = app2.post_json("/api/chat", {"session_id": "t3-02-b", "message": "second"})
        self.assertEqual(c2.status, 200)
        b2 = fake_llama.last_request["body"]
        self.assertAlmostEqual(b2["temperature"], 0.35)
        self.assertAlmostEqual(b2["top_p"], 0.75)
        self.assertEqual(b2["max_tokens"], 128)

    def test_t3_03_f1_f5_t2i_config_persists_across_restart(self) -> None:
        tmp = self._make_temp_dir("moris_t3_03_")
        cfg_path = tmp / "cfg.json"
        fake_llama = self._start_fake_llama(default_reply="重启保留T2I配置")

        app1 = AppProcessHarness(
            config_path=cfg_path,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            },
        ).start()
        try:
            put = app1.put_json(
                "/api/config",
                {
                    "t2i": {
                        "enabled": True,
                        "active_plugin": "example_stub",
                        "context_window_turns": 3,
                        "plugin_settings": {"ratio": "16:9"},
                    }
                },
            )
            self.assertEqual(put.status, 200)
        finally:
            app1.stop()

        app2 = self._start_app(config_path=cfg_path, write_config=False)
        ext = app2.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        self.assertIs(ext.data["enabled"], True)
        self.assertEqual(ext.data["active_plugin"], "example_stub")

        chat = app2.post_json("/api/chat", {"session_id": "t3-03", "message": "夜景"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["t2i_hook"]["status"], "stub_ok")
        self.assertIs(chat.data["t2i_hook"]["invoked"], True)

    def test_t3_04_f2_f4_firefly_skill_pinned_under_sliding_window(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"FireflyReply-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "max_history_turns": 2,
                }
            }
        )
        self.assertEqual(app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)}).status, 200)

        for i in range(1, 5):
            r = app.post_json("/api/chat", {"session_id": "ff-win", "message": f"TurnMsg-{i}"})
            self.assertEqual(r.status, 200)

        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("格拉默铁骑", msgs[0]["content"])
        self.assertIn("失熵症", msgs[0]["content"])
        self.assertNotIn("Copyright (c) 2026 Heart Ease", msgs[0]["content"])
        self.assertNotIn("evidence_summary", msgs[0]["content"])

        history_contents = [m["content"] for m in msgs[1:]]
        self.assertNotIn("TurnMsg-1", history_contents)
        self.assertEqual(
            history_contents,
            ["TurnMsg-2", "FireflyReply-2", "TurnMsg-3", "FireflyReply-3", "TurnMsg-4"],
        )

    def test_t3_05_f2_f5_skill_switch_updates_t2i_scene_context(self) -> None:
        tmp = self._make_temp_dir("moris_t3_05_")
        log_path = tmp / "switch_t2i.jsonl"
        silverwolf = make_single_md_skill(
            tmp,
            "silverwolf.md",
            "# SilverWolf\nMaster hacker of Stellaron Hunters.",
            {"name": "SilverWolf"},
        )
        fake_llama = self._start_fake_llama(default_reply="切换角色回复")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "switch_rec"},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "switch_rec.py",
            make_recording_plugin_source("switch_rec", log_path),
        )
        app.start()
        self.addCleanup(app.stop)

        app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
        self.assertEqual(app.post_json("/api/chat", {"session_id": "sw", "message": "橡木蛋糕卷"}).status, 200)

        app.post_json("/api/skill/load", {"path": str(silverwolf)})
        self.assertEqual(app.post_json("/api/chat", {"session_id": "sw", "message": "打游戏"}).status, 200)

        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["character_name"], "firefly-hsr")
        self.assertEqual(entries[1]["character_name"], "SilverWolf")
        self.assertIn("silverwolf.md", entries[1]["skill_source"])

    def test_t3_06_f3_f4_upstream_500_rollback_preserves_alternation(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"ok-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        self.assertEqual(app.post_json("/api/chat", {"session_id": "alt", "message": "M1"}).status, 200)

        fake_llama.enqueue_response(mode="http_500")
        err = app.post_json("/api/chat", {"session_id": "alt", "message": "M2-FAIL"})
        self.assertEqual(err.status, 502)

        ok = app.post_json("/api/chat", {"session_id": "alt", "message": "M3"}).status
        self.assertEqual(ok, 200)
        roles = [m["role"] for m in fake_llama.last_request["body"]["messages"] if m["role"] != "system"]
        self.assertEqual(roles, ["user", "assistant", "user"])
        contents = [m["content"] for m in fake_llama.last_request["body"]["messages"]]
        self.assertNotIn("M2-FAIL", contents)

    def test_t3_07_f4_f5_session_reset_clears_t2i_recent_messages_keeps_skill(self) -> None:
        tmp = self._make_temp_dir("moris_t3_07_")
        log_path = tmp / "rst_t2i.jsonl"
        skill = make_single_md_skill(tmp, "hero.md", "# Hero\nAlways brave.", {"name": "BraveHero"})
        fake_llama = self._start_fake_llama(default_reply="勇者回复")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "rst_rec", "context_window_turns": 4},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "rst_rec.py",
            make_recording_plugin_source("rst_rec", log_path),
        )
        app.start()
        self.addCleanup(app.stop)

        app.post_json("/api/skill/load", {"path": str(skill)})
        app.post_json("/api/chat", {"session_id": "r-s", "message": "PRE_RESET_1"})
        app.post_json("/api/chat", {"session_id": "r-s", "message": "PRE_RESET_2"})
        self.assertEqual(app.post_json("/api/sessions/reset", {"session_id": "r-s"}).status, 200)

        post = app.post_json("/api/chat", {"session_id": "r-s", "message": "POST_RESET"})
        self.assertEqual(post.status, 200)
        self.assertEqual(post.data["turn_index"], 1)

        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 3)
        last = entries[-1]
        self.assertEqual(last["character_name"], "BraveHero")
        self.assertEqual(last["turn_index"], 1)
        recent_dump = json.dumps(last["recent_messages"], ensure_ascii=False)
        self.assertNotIn("PRE_RESET", recent_dump)

    def test_t3_08_f4_f6_interleaved_sessions_and_selective_reset(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"reply-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        app.post_json("/api/chat", {"session_id": "alice", "message": "Alice-1"})
        app.post_json("/api/chat", {"session_id": "bob", "message": "Bob-1"})
        app.post_json("/api/sessions/reset", {"session_id": "alice"})

        a_post = app.post_json("/api/chat", {"session_id": "alice", "message": "Alice-New"})
        self.assertEqual(a_post.status, 200)
        self.assertEqual(a_post.data["turn_index"], 1)
        a_dump = json.dumps(fake_llama.last_request["body"]["messages"], ensure_ascii=False)
        self.assertNotIn("Alice-1", a_dump)
        self.assertNotIn("Bob-1", a_dump)

        b_post = app.post_json("/api/chat", {"session_id": "bob", "message": "Bob-2"})
        self.assertEqual(b_post.status, 200)
        self.assertEqual(b_post.data["turn_index"], 2)
        b_dump = json.dumps(fake_llama.last_request["body"]["messages"], ensure_ascii=False)
        self.assertIn("Bob-1", b_dump)
        self.assertNotIn("Alice", b_dump)

    def test_t3_09_f3_f6_engine_lifecycle_api_and_post_stop_chat_error(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="生命周期测试")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        self.assertEqual(app.get_json("/api/engine/status").status, 200)
        self.assertEqual(app.post_json("/api/engine/start", {}).data["state"], "ready")
        self.assertEqual(app.post_json("/api/chat", {"session_id": "lc", "message": "hi"}).status, 200)

        stop_r = app.post_json("/api/engine/stop", {})
        self.assertEqual(stop_r.status, 200)
        self.assertEqual(stop_r.data["state"], "stopped")
        self.assertEqual(app.get_json("/api/engine/status").data["state"], "stopped")

        # Stop FakeLlamaServer and verify post-stop chat/start failure returns structured error, never crash
        fake_llama.stop()
        err_chat = app.post_json("/api/chat", {"session_id": "lc", "message": "after stop"})
        self.assertIn(err_chat.status, (400, 502, 503))
        self.assertIn("error", err_chat.data)
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_t3_10_f6_f7_start_bat_static_offline_ui_and_clean_port_release(self) -> None:
        app = AppProcessHarness(use_bat_launcher=True).start()
        port = app.port
        try:
            html = app.get_text("/")
            css = app.get_text("/static/style.css")
            js = app.get_text("/static/app.js")
            self.assertEqual(html.status, 200)
            self.assertEqual(css.status, 200)
            self.assertEqual(js.status, 200)
            self.assertIsNone(re.search(r"""(?:src|href)\s*=\s*["']https?://""", html.text, re.I))
            self.assertEqual(app.get_json("/api/health").status, 200)
        finally:
            app.stop()

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and is_port_open("127.0.0.1", port):
            time.sleep(0.1)
        self.assertFalse(is_port_open("127.0.0.1", port))

    def test_t3_11_f1_f7_cli_config_autocreation_and_port_override(self) -> None:
        tmp = self._make_temp_dir("moris_t3_11_")
        cfg_path = tmp / "fresh_cli_cfg.json"
        cli_port = find_free_port()
        app = self._start_app(config_path=cfg_path, write_config=False, port=cli_port)
        self.assertTrue(cfg_path.is_file())
        self.assertEqual(app.get_json("/api/health").status, 200)
        cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertIn("server", cfg_data)
        self.assertIn("llama", cfg_data)

    def test_t3_12_f5_f8_f2_documented_extensibility_contract_end_to_end(self) -> None:
        readme = require_target_file(REPO_ROOT / "README.md").read_text(encoding="utf-8")
        ext_doc = require_target_file(REPO_ROOT / "docs" / "EXTENSIONS.md").read_text(encoding="utf-8")
        self.assertIn("start.bat", readme)
        self.assertIn("SceneContext", ext_doc)
        self.assertIn("T2IResult", ext_doc)
        self.assertIn(".md", ext_doc)

        tmp = self._make_temp_dir("moris_t3_12_")
        log_path = tmp / "doc_plugin.jsonl"
        skill_dir = tmp / "doc_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: DocHero\n---\n# Doc Hero", encoding="utf-8")
        (skill_dir / "ignore_me.txt").write_text("IGNORE_TOKEN", encoding="utf-8")

        fake_llama = self._start_fake_llama(default_reply="文档契约验证成功")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "doc_stub"},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "doc_stub.py",
            make_recording_plugin_source("doc_stub", log_path, use_register=True),
        )
        app.start()
        self.addCleanup(app.stop)

        load = app.post_json("/api/skill/load", {"path": str(skill_dir)})
        self.assertEqual(load.status, 200)
        self.assertIn("ignore_me.txt", load.data["ignored_files"])
        self.assertNotIn("IGNORE_TOKEN", load.data["system_prompt"])

        chat = app.post_json("/api/chat", {"session_id": "doc-e2e", "message": "按文档扩展"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["t2i_hook"]["plugin_name"], "doc_stub")
        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["character_name"], "DocHero")


if __name__ == "__main__":
    unittest.main()
