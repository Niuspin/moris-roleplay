"""Tier 1 E2E tests: Single-feature happy paths across F1-F8 (48 tests, zero skips)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from e2e_tests.harness import (
    AEMEATH_SKILL_DIR,
    AppProcessHarness,
    FIREFLY_SKILL_DIR,
    FakeLlamaServer,
    RAIDEN_SKILL_FILE,
    REPO_ROOT,
    SANDRONE_SKILL_FILE,
    YAE_MIKO_SKILL_FILE,
    build_mock_llama_exe,
    find_browser_executable,
    find_free_port,
    is_pid_alive,
    is_port_open,
    make_recording_plugin_source,
    make_single_md_skill,
    read_jsonl,
    read_mock_llama_args_log,
    require_target_file,
    run_headless_edge_dom,
    write_dummy_gguf,
    write_t2i_plugin,
)


class _Tier1Base(unittest.TestCase):
    def _make_temp_dir(self, prefix: str = "moris_t1_") -> Path:
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


class TestTier1F1Config(_Tier1Base):
    """F1: Atomic Config Store (6 Tier 1 tests)."""

    def test_f1_t1_01_config_example_schema_and_safe_defaults(self) -> None:
        cfg_file = require_target_file(REPO_ROOT / "config.example.json")
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        for section in ("server", "skill", "llama", "t2i"):
            self.assertIn(section, data)
        self.assertEqual(data["server"]["host"], "127.0.0.1")
        self.assertEqual(data["server"]["port"], 7860)
        self.assertIs(data["server"]["open_browser"], True)
        self.assertIn("last_path", data["skill"])
        self.assertIn("exclude_meta_docs", data["skill"])
        self.assertIn("max_prompt_chars", data["skill"])
        self.assertEqual(data["llama"]["engine_mode"], "managed_server")
        self.assertIn("ctx_size", data["llama"])
        self.assertIn("max_history_turns", data["llama"])
        self.assertIs(data["t2i"]["enabled"], False)
        self.assertEqual(data["t2i"]["active_plugin"], "noop")
        self.assertEqual(data["t2i"]["context_window_turns"], 4)
        self.assertEqual(data["t2i"]["plugins_dir"], "plugins/t2i")
        self.assertIsInstance(data["t2i"]["plugin_settings"], dict)

    def test_f1_t1_02_auto_creation_on_missing_config(self) -> None:
        tmp = self._make_temp_dir("moris_f1_02_")
        missing_cfg = tmp / "auto_created" / "config.json"
        app = self._start_app(config_path=missing_cfg, write_config=False)

        health = app.get_json("/api/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(health.data["status"], "ok")
        self.assertTrue(missing_cfg.is_file())

        cfg_resp = app.get_json("/api/config")
        self.assertEqual(cfg_resp.status, 200)
        for section in ("server", "skill", "llama", "t2i"):
            self.assertIn(section, cfg_resp.data)
        self.assertIs(cfg_resp.data["t2i"]["enabled"], False)
        self.assertEqual(cfg_resp.data["t2i"]["active_plugin"], "noop")

    def test_f1_t1_03_deep_merge_partial_put_config(self) -> None:
        app = self._start_app()
        before = app.get_json("/api/config").data
        orig_ctx = before["llama"]["ctx_size"]
        orig_port = before["server"]["port"]

        put_resp = app.put_json(
            "/api/config",
            {"llama": {"temperature": 0.45, "max_tokens": 256}},
        )
        self.assertEqual(put_resp.status, 200)
        self.assertAlmostEqual(put_resp.data["llama"]["temperature"], 0.45)
        self.assertEqual(put_resp.data["llama"]["max_tokens"], 256)
        self.assertEqual(put_resp.data["llama"]["ctx_size"], orig_ctx)
        self.assertEqual(put_resp.data["server"]["port"], orig_port)
        self.assertIs(put_resp.data["t2i"]["enabled"], False)

    def test_f1_t1_04_persistence_across_restart_unicode_paths(self) -> None:
        tmp = self._make_temp_dir("moris_f1_04_")
        cfg_path = tmp / "persist_config.json"
        app1 = AppProcessHarness(config_path=cfg_path).start()
        try:
            resp = app1.put_json(
                "/api/config",
                {
                    "skill": {"last_path": r"C:\角色\流萤"},
                    "llama": {"model_path": r"C:\模型\demo.gguf"},
                },
            )
            self.assertEqual(resp.status, 200)
        finally:
            app1.stop()

        app2 = self._start_app(config_path=cfg_path, write_config=False)
        cfg2 = app2.get_json("/api/config")
        self.assertEqual(cfg2.status, 200)
        self.assertEqual(cfg2.data["skill"]["last_path"], r"C:\角色\流萤")
        self.assertEqual(cfg2.data["llama"]["model_path"], r"C:\模型\demo.gguf")
        disk_json = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_json["skill"]["last_path"], r"C:\角色\流萤")

    def test_f1_t1_05_multi_section_update_and_atomic_disk_write(self) -> None:
        tmp = self._make_temp_dir("moris_f1_05_")
        cfg_path = tmp / "atomic.json"
        app = self._start_app(config_path=cfg_path)
        patch = {
            "server": {"open_browser": False},
            "skill": {"exclude_meta_docs": True, "max_prompt_chars": 12000},
            "llama": {"max_history_turns": 12, "top_p": 0.85},
            "t2i": {"context_window_turns": 2},
        }
        resp = app.put_json("/api/config", patch)
        self.assertEqual(resp.status, 200)
        disk_data = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(disk_data["skill"]["exclude_meta_docs"], True)
        self.assertEqual(disk_data["skill"]["max_prompt_chars"], 12000)
        self.assertEqual(disk_data["llama"]["max_history_turns"], 12)
        self.assertEqual(disk_data["t2i"]["context_window_turns"], 2)
        leftovers = [p.name for p in cfg_path.parent.iterdir() if p.name != "atomic.json"]
        self.assertEqual(leftovers, [])

    def test_f1_t1_06_t2i_plugin_settings_nested_roundtrip(self) -> None:
        app = self._start_app()
        nested = {
            "model": "sdxl-turbo",
            "steps": 24,
            "cfg_scale": 7.5,
            "options": {"sampler": "euler_a", "tags": ["firefly", "starlight"]},
        }
        put_resp = app.put_json("/api/config", {"t2i": {"plugin_settings": nested}})
        self.assertEqual(put_resp.status, 200)
        get_resp = app.get_json("/api/config")
        self.assertEqual(get_resp.status, 200)
        self.assertEqual(get_resp.data["t2i"]["plugin_settings"], nested)


class TestTier1F2SkillLoader(_Tier1Base):
    """F2: .md-Only Skill Loader (7 Tier 1 tests)."""

    def test_f2_t1_01_load_real_firefly_dir_all_md(self) -> None:
        app = self._start_app()
        resp = app.post_json(
            "/api/skill/load",
            {"path": str(FIREFLY_SKILL_DIR), "exclude_meta_docs": False},
        )
        self.assertEqual(resp.status, 200)
        data = resp.data
        self.assertEqual(data["source_type"], "directory")
        self.assertEqual(data["character_name"], "firefly-hsr")
        expected_order = [
            "SKILL.md",
            "prompt.md",
            "profile.md",
            "personality.md",
            "interaction.md",
            "relations.md",
            "conflicts.md",
            "memory.md",
            "README.md",
            "quality-report.md",
            "test-report.md",
        ]
        self.assertEqual(data["loaded_files"], expected_order)
        self.assertIn("LICENSE", data["ignored_files"])
        self.assertIn("manifest.json", data["ignored_files"])
        self.assertEqual(data["excluded_files"], [])
        for entry in data["loaded_files"] + data["ignored_files"]:
            self.assertNotIn(".git", entry)

        prompt = data["system_prompt"]
        canonical_tokens = [
            "绝对不可违背的最高规则",
            "时装“春日手信”",
            "# 流萤 - 角色档案",
            "# 流萤 - 性格与价值观",
            "# 流萤 - 说话方式与互动风格",
            "# 流萤 - 人际关系网络",
            "# 设定冲突记录",
            "# 流萤 - 背景故事与关键记忆",
            "# 流萤.skill",
            "# 质量评分报告：流萤",
            "# 扮演测试报告：流萤",
        ]
        indices = [prompt.index(tok) for tok in canonical_tokens]
        self.assertEqual(indices, sorted(indices))

    def test_f2_t1_02_firefly_ignores_non_md_and_strips_frontmatter(self) -> None:
        app = self._start_app()
        resp = app.post_json(
            "/api/skill/load",
            {"path": str(FIREFLY_SKILL_DIR), "exclude_meta_docs": False},
        )
        self.assertEqual(resp.status, 200)
        prompt = resp.data["system_prompt"]

        for forbidden in (
            "Copyright (c) 2026 Heart Ease",
            "WITHOUT WARRANTY OF ANY KIND",
            "evidence_summary",
            '"built_at": "2026-04-10T12:25:00+08:00"',
            '"auto_activate": true',
            "ref: refs/heads/main",
            "pack-395cdff8",
            "name: firefly-hsr",
            'metadata: {"kit":"character-skill"',
        ):
            self.assertNotIn(forbidden, prompt)

        self.assertNotIn("\r", prompt)
        # conflicts.md internal horizontal rules must remain intact
        self.assertGreaterEqual(prompt.count("\n---\n"), 3)

    def test_f2_t1_03_firefly_exclude_meta_docs_true(self) -> None:
        app = self._start_app()
        full_resp = app.post_json(
            "/api/skill/load",
            {"path": str(FIREFLY_SKILL_DIR), "exclude_meta_docs": False},
        )
        self.assertEqual(full_resp.status, 200)

        excl_resp = app.post_json(
            "/api/skill/load",
            {"path": str(FIREFLY_SKILL_DIR), "exclude_meta_docs": True},
        )
        self.assertEqual(excl_resp.status, 200)
        data = excl_resp.data
        self.assertEqual(
            data["loaded_files"],
            [
                "SKILL.md",
                "prompt.md",
                "profile.md",
                "personality.md",
                "interaction.md",
                "relations.md",
                "conflicts.md",
                "memory.md",
            ],
        )
        self.assertEqual(
            sorted(data["excluded_files"]),
            sorted(["README.md", "quality-report.md", "test-report.md"]),
        )
        self.assertNotIn("# 流萤.skill", data["system_prompt"])
        self.assertNotIn("0.937 / 1.0", data["system_prompt"])
        self.assertLess(data["total_chars"], full_resp.data["total_chars"])

    def test_f2_t1_04_load_real_single_file_yae_miko(self) -> None:
        app = self._start_app()
        resp = app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)})
        self.assertEqual(resp.status, 200)
        data = resp.data
        self.assertEqual(data["source_type"], "file")
        self.assertEqual(data["character_name"], "yae-miko-perspective")
        self.assertIn("鸣神大社宫司", data["description"])
        self.assertEqual(data["loaded_files"], ["SKILL.md"])
        self.assertIn("# 八重神子 · 思维操作系统", data["system_prompt"])
        self.assertFalse(data["system_prompt"].lstrip().startswith("---"))
        self.assertNotIn("name: yae-miko-perspective", data["system_prompt"])

    def test_f2_t1_05_load_other_real_skills_sandrone_raiden_aemeath(self) -> None:
        app = self._start_app()

        r1 = app.post_json("/api/skill/load", {"path": str(SANDRONE_SKILL_FILE)})
        self.assertEqual(r1.status, 200)
        self.assertEqual(r1.data["character_name"], "sandrone-perspective")

        r2 = app.post_json("/api/skill/load", {"path": str(RAIDEN_SKILL_FILE)})
        self.assertEqual(r2.status, 200)
        self.assertEqual(r2.data["character_name"], "raiden-shogun-perspective")

        r3 = app.post_json("/api/skill/load", {"path": str(AEMEATH_SKILL_DIR)})
        self.assertEqual(r3.status, 200)
        self.assertEqual(r3.data["character_name"], "aemeath")
        self.assertIn("LICENSE", r3.data["ignored_files"])
        self.assertIn("manifest.json", r3.data["ignored_files"])
        self.assertNotIn("# 爱弥斯行为蒸馏链", r3.data["system_prompt"])

        curr = app.get_json("/api/skill/current")
        self.assertEqual(curr.status, 200)
        self.assertEqual(curr.data["character_name"], "aemeath")

    def test_f2_t1_06_synthetic_mixed_extensions_and_uppercase_md(self) -> None:
        tmp = self._make_temp_dir("moris_f2_06_")
        skill_dir = tmp / "mixed_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: MixedHero\n---\n# Hero Core", encoding="utf-8")
        (skill_dir / "alpha.md").write_text("Alpha rule", encoding="utf-8")
        (skill_dir / "extra.MD").write_text("Uppercase MD rule", encoding="utf-8")
        (skill_dir / "notes.txt").write_text("SECRET_TXT_SHOULD_NOT_LOAD", encoding="utf-8")
        (skill_dir / "data.json").write_text('{"secret_json": true}', encoding="utf-8")
        (skill_dir / "run.bat").write_text("@echo off\n", encoding="utf-8")
        (skill_dir / "avatar.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\xff\xfe")
        hidden = skill_dir / ".hidden"
        hidden.mkdir()
        (hidden / "secret.md").write_text("HIDDEN_DOT_DIR_SECRET", encoding="utf-8")

        app = self._start_app()
        resp = app.post_json("/api/skill/load", {"path": str(skill_dir)})
        self.assertEqual(resp.status, 200)
        data = resp.data
        self.assertEqual(data["loaded_files"], ["SKILL.md", "alpha.md", "extra.MD"])
        for non_md in ("notes.txt", "data.json", "run.bat", "avatar.png"):
            self.assertIn(non_md, data["ignored_files"])
        self.assertIn("Uppercase MD rule", data["system_prompt"])
        self.assertNotIn("SECRET_TXT_SHOULD_NOT_LOAD", data["system_prompt"])
        self.assertNotIn("secret_json", data["system_prompt"])
        self.assertNotIn("HIDDEN_DOT_DIR_SECRET", data["system_prompt"])

    def test_f2_t1_07_utf8_bom_crlf_and_skill_persistence_across_chat(self) -> None:
        tmp = self._make_temp_dir("moris_f2_07_")
        skill_file = make_single_md_skill(
            tmp,
            "custom_hero.md",
            "# Custom Hero\nSpeak warmly.",
            frontmatter=None,
            encoding="utf-8-sig",
            newline="\r\n",
        )
        fake_llama = self._start_fake_llama(default_reply="你好，旅行者。")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        load_resp = app.post_json("/api/skill/load", {"path": str(skill_file)})
        self.assertEqual(load_resp.status, 200)
        self.assertEqual(load_resp.data["character_name"], "custom_hero")
        self.assertNotIn("\ufeff", load_resp.data["system_prompt"])
        self.assertNotIn("\r", load_resp.data["system_prompt"])

        curr = app.get_json("/api/skill/current")
        self.assertEqual(curr.status, 200)
        self.assertEqual(curr.data["character_name"], "custom_hero")

        chat = app.post_json("/api/chat", {"session_id": "s-bom", "message": "Hi"})
        self.assertEqual(chat.status, 200)
        sent_msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(sent_msgs[0]["role"], "system")
        self.assertEqual(sent_msgs[0]["content"], load_resp.data["system_prompt"])


class TestTier1F3LlamaEngine(_Tier1Base):
    """F3: llama.cpp Engine Adapter (7 Tier 1 tests)."""

    def test_f3_t1_01_external_server_start_health_and_chat(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="外部服务回复成功。")
        app = self._start_app()
        put_resp = app.put_json(
            "/api/config",
            {"llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}},
        )
        self.assertEqual(put_resp.status, 200)

        start_resp = app.post_json("/api/engine/start", {})
        self.assertEqual(start_resp.status, 200)
        self.assertEqual(start_resp.data["state"], "ready")
        self.assertEqual(start_resp.data["mode"], "external_server")

        status_resp = app.get_json("/api/engine/status")
        self.assertEqual(status_resp.status, 200)
        self.assertEqual(status_resp.data["state"], "ready")

        health = app.get_json("/api/health")
        self.assertEqual(health.status, 200)
        self.assertEqual(health.data["engine"]["state"], "ready")

        chat = app.post_json("/api/chat", {"session_id": "ext-1", "message": "测试连接"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["reply"], "外部服务回复成功。")

    def test_f3_t1_02_managed_server_exe_path_and_cli_arg_forwarding(self) -> None:
        tmp = self._make_temp_dir("moris_f3_02_")
        exe_path = build_mock_llama_exe(tmp / "bin", "llama-server.exe")
        gguf_path = write_dummy_gguf(tmp / "models" / "tiny.gguf")
        args_log = tmp / "args.jsonl"
        llama_port = find_free_port()

        app = self._start_app(
            env_overrides={"MOCK_LLAMA_ARGS_LOG": str(args_log)},
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe_path),
                    "model_path": str(gguf_path),
                    "port": llama_port,
                    "ctx_size": 4096,
                    "n_gpu_layers": 16,
                    "startup_timeout_sec": 10,
                }
            },
        )
        start_resp = app.post_json("/api/engine/start", {})
        self.assertEqual(start_resp.status, 200)
        self.assertEqual(start_resp.data["state"], "ready")
        self.assertGreater(int(start_resp.data["pid"]), 0)

        logged = read_mock_llama_args_log(args_log)
        self.assertEqual(len(logged), 1)
        argv = logged[0]
        self.assertIn("-m", argv)
        self.assertIn(str(gguf_path), argv)
        self.assertIn("--host", argv)
        self.assertIn("127.0.0.1", argv)
        self.assertIn("--port", argv)
        self.assertIn(str(llama_port), argv)
        self.assertIn("-c", argv)
        self.assertIn("4096", argv)
        self.assertIn("-ngl", argv)
        self.assertIn("16", argv)

        chat = app.post_json("/api/chat", {"session_id": "m-1", "message": "Hello"})
        self.assertEqual(chat.status, 200)
        self.assertIn("Mock C# llama-server reply", chat.data["reply"])

    def test_f3_t1_03_managed_server_directory_discovery_llama_server_and_server_exe(self) -> None:
        tmp = self._make_temp_dir("moris_f3_03_")
        dir1 = tmp / "dir_llama"
        dir2 = tmp / "dir_server"
        build_mock_llama_exe(dir1, "llama-server.exe")
        build_mock_llama_exe(dir2, "server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")

        app = self._start_app()
        for idx, binary_dir in enumerate((dir1, dir2), start=1):
            p = find_free_port()
            put = app.put_json(
                "/api/config",
                {
                    "llama": {
                        "engine_mode": "managed_server",
                        "llama_cpp_path": str(binary_dir),
                        "model_path": str(gguf),
                        "port": p,
                        "startup_timeout_sec": 10,
                    }
                },
            )
            self.assertEqual(put.status, 200)
            st = app.post_json("/api/engine/start", {})
            self.assertEqual(st.status, 200, f"Failed for dir {idx}: {st.text}")
            self.assertEqual(st.data["state"], "ready")
            stop = app.post_json("/api/engine/stop", {})
            self.assertEqual(stop.status, 200)

    def test_f3_t1_04_managed_server_stop_and_port_release(self) -> None:
        tmp = self._make_temp_dir("moris_f3_04_")
        exe = build_mock_llama_exe(tmp, "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        llama_port = find_free_port()

        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": llama_port,
                    "startup_timeout_sec": 10,
                }
            }
        )
        start_resp = app.post_json("/api/engine/start", {})
        self.assertEqual(start_resp.status, 200)
        pid = int(start_resp.data["pid"])
        self.assertTrue(is_pid_alive(pid))
        self.assertTrue(is_port_open("127.0.0.1", llama_port))

        stop_resp = app.post_json("/api/engine/stop", {})
        self.assertEqual(stop_resp.status, 200)
        self.assertEqual(stop_resp.data["state"], "stopped")

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and (is_pid_alive(pid) or is_port_open("127.0.0.1", llama_port)):
            time.sleep(0.1)
        self.assertFalse(is_pid_alive(pid))
        self.assertFalse(is_port_open("127.0.0.1", llama_port))

    def test_f3_t1_05_app_shutdown_cleans_up_managed_llama_server(self) -> None:
        tmp = self._make_temp_dir("moris_f3_05_")
        exe = build_mock_llama_exe(tmp, "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        llama_port = find_free_port()

        app = AppProcessHarness(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": llama_port,
                    "startup_timeout_sec": 10,
                }
            }
        ).start()
        try:
            st = app.post_json("/api/engine/start", {})
            self.assertEqual(st.status, 200)
            managed_pid = int(st.data["pid"])
            self.assertTrue(is_pid_alive(managed_pid))
            app.send_ctrl_break()
            if app.proc is not None:
                try:
                    app.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            app.stop()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and is_pid_alive(managed_pid):
            time.sleep(0.1)
        self.assertFalse(is_pid_alive(managed_pid))
        self.assertFalse(is_port_open("127.0.0.1", llama_port))

    def test_f3_t1_06_loading_503_recovers_to_ready_during_startup_poll(self) -> None:
        tmp = self._make_temp_dir("moris_f3_06_")
        exe = build_mock_llama_exe(tmp, "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        llama_port = find_free_port()

        app = self._start_app(
            env_overrides={"MOCK_LLAMA_MODE": "loading_then_ready:3"},
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": llama_port,
                    "startup_timeout_sec": 10,
                }
            },
        )
        st = app.post_json("/api/engine/start", {})
        self.assertEqual(st.status, 200)
        self.assertEqual(st.data["state"], "ready")

    def test_f3_t1_07_switch_between_managed_and_external_mode(self) -> None:
        tmp = self._make_temp_dir("moris_f3_07_")
        exe = build_mock_llama_exe(tmp, "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        fake_ext = self._start_fake_llama(default_reply="已切换为外部服务。")

        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": find_free_port(),
                    "startup_timeout_sec": 10,
                }
            }
        )
        st1 = app.post_json("/api/engine/start", {})
        self.assertEqual(st1.status, 200)
        managed_pid = int(st1.data["pid"])

        put = app.put_json(
            "/api/config",
            {"llama": {"engine_mode": "external_server", "server_url": fake_ext.base_url}},
        )
        self.assertEqual(put.status, 200)
        st2 = app.post_json("/api/engine/start", {})
        self.assertEqual(st2.status, 200)
        self.assertEqual(st2.data["mode"], "external_server")
        self.assertEqual(st2.data["state"], "ready")

        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and is_pid_alive(managed_pid):
            time.sleep(0.1)
        self.assertFalse(is_pid_alive(managed_pid))

        chat = app.post_json("/api/chat", {"session_id": "sw-1", "message": "Hi"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["reply"], "已切换为外部服务。")


class TestTier1F4MultiTurnChat(_Tier1Base):
    """F4: Multi-Turn Chat Service (6 Tier 1 tests)."""

    def test_f4_t1_01_skill_system_prompt_injected_turn1(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="开拓者，很高兴见到你！")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        skill_resp = app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
        self.assertEqual(skill_resp.status, 200)
        expected_sys = skill_resp.data["system_prompt"]

        chat = app.post_json("/api/chat", {"session_id": "s1", "message": "你好，流萤"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["session_id"], "s1")
        self.assertEqual(chat.data["turn_index"], 1)
        self.assertEqual(chat.data["reply"], "开拓者，很高兴见到你！")

        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": expected_sys})
        self.assertEqual(msgs[1], {"role": "user", "content": "你好，流萤"})

    def test_f4_t1_02_multi_turn_history_chronological_preservation(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"A{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        tmp = self._make_temp_dir("moris_f4_02_")
        s_path = make_single_md_skill(tmp, "hero.md", "# Hero\nAlways concise.", {"name": "Hero"})
        self.assertEqual(app.post_json("/api/skill/load", {"path": str(s_path)}).status, 200)

        r1 = app.post_json("/api/chat", {"session_id": "s-multi", "message": "U1"})
        r2 = app.post_json("/api/chat", {"session_id": "s-multi", "message": "U2"})
        r3 = app.post_json("/api/chat", {"session_id": "s-multi", "message": "U3"})
        self.assertEqual([r1.data["turn_index"], r2.data["turn_index"], r3.data["turn_index"]], [1, 2, 3])
        self.assertLess(r1.data["messages_count"], r2.data["messages_count"])
        self.assertLess(r2.data["messages_count"], r3.data["messages_count"])

        turn3_msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(
            [(m["role"], m["content"]) for m in turn3_msgs[1:]],
            [
                ("user", "U1"),
                ("assistant", "A1"),
                ("user", "U2"),
                ("assistant", "A2"),
                ("user", "U3"),
            ],
        )

    def test_f4_t1_03_sliding_window_max_history_turns(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"R{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "max_history_turns": 2,
                }
            }
        )
        tmp = self._make_temp_dir("moris_f4_03_")
        s_path = make_single_md_skill(tmp, "pinned.md", "PINNED_SYSTEM_RULE", {"name": "Pinned"})
        app.post_json("/api/skill/load", {"path": str(s_path)})

        for n in range(1, 5):
            resp = app.post_json("/api/chat", {"session_id": "s-win", "message": f"Q{n}"})
            self.assertEqual(resp.status, 200)

        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("PINNED_SYSTEM_RULE", msgs[0]["content"])
        self.assertEqual(
            [(m["role"], m["content"]) for m in msgs[1:]],
            [
                ("user", "Q2"),
                ("assistant", "R2"),
                ("user", "Q3"),
                ("assistant", "R3"),
                ("user", "Q4"),
            ],
        )

    def test_f4_t1_04_session_isolation_multiple_sessions(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"reply-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )

        self.assertEqual(
            app.post_json("/api/chat", {"session_id": "sess-alpha", "message": "秘密Alpha"}).status,
            200,
        )
        self.assertEqual(
            app.post_json("/api/chat", {"session_id": "sess-beta", "message": "秘密Beta"}).status,
            200,
        )
        self.assertEqual(
            app.post_json("/api/chat", {"session_id": "sess-alpha", "message": "Alpha续问"}).status,
            200,
        )
        self.assertEqual(
            app.post_json("/api/chat", {"session_id": "sess-beta", "message": "Beta续问"}).status,
            200,
        )

        reqs = fake_llama.requests
        self.assertEqual(len(reqs), 4)
        alpha_dump = json.dumps(reqs[2]["body"]["messages"], ensure_ascii=False)
        beta_dump = json.dumps(reqs[3]["body"]["messages"], ensure_ascii=False)
        self.assertIn("秘密Alpha", alpha_dump)
        self.assertNotIn("秘密Beta", alpha_dump)
        self.assertIn("秘密Beta", beta_dump)
        self.assertNotIn("秘密Alpha", beta_dump)

    def test_f4_t1_05_session_reset_clears_turns_keeps_skill(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"resp-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        tmp = self._make_temp_dir("moris_f4_05_")
        s_path = make_single_md_skill(tmp, "keep.md", "KEEP_THIS_SKILL", {"name": "KeepChar"})
        app.post_json("/api/skill/load", {"path": str(s_path)})

        app.post_json("/api/chat", {"session_id": "s-reset", "message": "old-1"})
        app.post_json("/api/chat", {"session_id": "s-reset", "message": "old-2"})
        app.post_json("/api/chat", {"session_id": "s-other", "message": "other-1"})

        rst = app.post_json("/api/sessions/reset", {"session_id": "s-reset"})
        self.assertEqual(rst.status, 200)

        curr = app.get_json("/api/skill/current")
        self.assertEqual(curr.status, 200)
        self.assertEqual(curr.data["character_name"], "KeepChar")

        after = app.post_json("/api/chat", {"session_id": "s-reset", "message": "fresh-1"})
        self.assertEqual(after.status, 200)
        self.assertEqual(after.data["turn_index"], 1)
        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertIn("KEEP_THIS_SKILL", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "fresh-1")

        other_t2 = app.post_json("/api/chat", {"session_id": "s-other", "message": "other-2"})
        self.assertEqual(other_t2.status, 200)
        self.assertEqual(other_t2.data["turn_index"], 2)
        other_dump = json.dumps(fake_llama.last_request["body"]["messages"], ensure_ascii=False)
        self.assertIn("other-1", other_dump)

    def test_f4_t1_06_sampling_params_and_skill_hot_swap(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="好的")
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "temperature": 0.35,
                    "top_p": 0.75,
                    "max_tokens": 256,
                }
            }
        )
        tmp = self._make_temp_dir("moris_f4_06_")
        s1 = make_single_md_skill(tmp, "char1.md", "CHAR_ONE_PROMPT", {"name": "CharOne"})
        app.post_json("/api/skill/load", {"path": str(s1)})
        self.assertEqual(app.post_json("/api/chat", {"session_id": "swap", "message": "hi"}).status, 200)

        req1 = fake_llama.last_request["body"]
        self.assertAlmostEqual(req1["temperature"], 0.35)
        self.assertAlmostEqual(req1["top_p"], 0.75)
        self.assertEqual(req1["max_tokens"], 256)
        self.assertIn("CHAR_ONE_PROMPT", req1["messages"][0]["content"])

        self.assertEqual(app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)}).status, 200)
        self.assertEqual(app.post_json("/api/chat", {"session_id": "swap", "message": "next"}).status, 200)
        req2 = fake_llama.last_request["body"]
        self.assertIn("# 八重神子 · 思维操作系统", req2["messages"][0]["content"])


class TestTier1F5T2IExtension(_Tier1Base):
    """F5: Post-Turn T2I Extension Interface (6 Tier 1 tests)."""

    def test_f5_t1_01_default_disabled_noop_on_chat(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="默认不触发文生图。")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        self.assertIs(ext.data["enabled"], False)
        self.assertEqual(ext.data["active_plugin"], "noop")
        plugin_names = {p["name"] for p in ext.data["plugins"]}
        self.assertIn("noop", plugin_names)
        self.assertIn("example_stub", plugin_names)

        chat = app.post_json("/api/chat", {"session_id": "t2i-def", "message": "看夜景"})
        self.assertEqual(chat.status, 200)
        hook = chat.data["t2i_hook"]
        self.assertEqual(hook["status"], "disabled")
        self.assertIs(hook["invoked"], False)
        self.assertFalse(hook.get("image_url"))
        self.assertFalse(hook.get("image_path"))

    def test_f5_t1_02_enable_builtin_example_stub_via_put_config(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="天台的晚风很温柔。")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        put = app.put_json(
            "/api/config",
            {
                "t2i": {
                    "enabled": True,
                    "active_plugin": "example_stub",
                    "plugin_settings": {"style": "anime"},
                }
            },
        )
        self.assertEqual(put.status, 200)
        ext = app.get_json("/api/extensions/t2i")
        self.assertIs(ext.data["enabled"], True)
        self.assertEqual(ext.data["active_plugin"], "example_stub")

        app.post_json("/api/skill/load", {"path": str(FIREFLY_SKILL_DIR)})
        chat = app.post_json("/api/chat", {"session_id": "t2i-ex", "message": "一起看星空"})
        self.assertEqual(chat.status, 200)
        hook = chat.data["t2i_hook"]
        self.assertEqual(hook["status"], "stub_ok")
        self.assertIs(hook["invoked"], True)
        self.assertEqual(hook["plugin_name"], "example_stub")
        self.assertTrue(hook["prompt_used"])

    def test_f5_t1_03_dropin_plugin_export_receives_full_scene_context(self) -> None:
        tmp = self._make_temp_dir("moris_f5_03_")
        log_path = tmp / "scene_log.jsonl"
        skill_file = make_single_md_skill(tmp, "aurora.md", "# Aurora\nGentle astronomer.", {"name": "Aurora"})
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"Aurora reply {idx}")

        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {
                    "enabled": True,
                    "active_plugin": "recording_stub",
                    "context_window_turns": 2,
                    "plugin_settings": {"seed": 42},
                },
            },
        )
        self.addCleanup(app.stop)
        assert app.temp_dir.exists()
        plugins_dir = app.temp_dir / "plugins" / "t2i"
        write_t2i_plugin(
            plugins_dir,
            "recording_stub.py",
            make_recording_plugin_source("recording_stub", log_path, use_register=False),
        )
        app.start()

        ext = app.get_json("/api/extensions/t2i")
        self.assertIn("recording_stub", {p["name"] for p in ext.data["plugins"]})
        app.post_json("/api/skill/load", {"path": str(skill_file)})

        for idx in range(1, 4):
            resp = app.post_json(
                "/api/chat",
                {"session_id": "ctx-sess", "message": f"Star prompt {idx}"},
            )
            self.assertEqual(resp.status, 200)
            hook = resp.data["t2i_hook"]
            self.assertEqual(hook["status"], "stub_ok")
            self.assertIn("elapsed_ms", hook)
            self.assertIsInstance(hook["elapsed_ms"], (int, float))
            self.assertIn("extra", hook)
            self.assertIsInstance(hook["extra"], dict)

        entries = read_jsonl(log_path)
        self.assertEqual(len(entries), 3)
        last = entries[-1]
        self.assertEqual(last["session_id"], "ctx-sess")
        self.assertEqual(last["turn_index"], 3)
        self.assertEqual(last["character_name"], "Aurora")
        self.assertTrue(last["timestamp_utc"])
        self.assertIn("aurora.md", last["skill_source"])
        self.assertEqual(last["user_message"], "Star prompt 3")
        self.assertEqual(last["assistant_reply"], "Aurora reply 3")
        self.assertLessEqual(len(last["recent_messages"]), 4)
        self.assertIn("scene_prompt_hint", last)
        self.assertIn("metadata", last)
        self.assertIsInstance(last["metadata"], dict)
        self.assertEqual(last["plugin_settings"], {"seed": 42})

    def test_f5_t1_04_dropin_register_function_discovery(self) -> None:
        tmp = self._make_temp_dir("moris_f5_04_")
        log_path = tmp / "reg_log.jsonl"
        fake_llama = self._start_fake_llama(default_reply="注册函数测试回复")

        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "reg_plugin"},
            },
        )
        self.addCleanup(app.stop)
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "reg_plugin.py",
            make_recording_plugin_source("reg_plugin", log_path, use_register=True),
        )
        app.start()

        ext = app.get_json("/api/extensions/t2i")
        self.assertIn("reg_plugin", {p["name"] for p in ext.data["plugins"]})
        chat = app.post_json("/api/chat", {"session_id": "reg-s", "message": "画一幅画"})
        self.assertEqual(chat.status, 200)
        self.assertIs(chat.data["t2i_hook"]["invoked"], True)
        self.assertEqual(chat.data["t2i_hook"]["status"], "stub_ok")
        self.assertEqual(len(read_jsonl(log_path)), 1)

    def test_f5_t1_05_dynamic_toggle_off_on_off_without_restart(self) -> None:
        tmp = self._make_temp_dir("moris_f5_05_")
        log_path = tmp / "toggle_log.jsonl"
        fake_llama = self._start_fake_llama(default_reply="切换回复")

        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": False, "active_plugin": "recording_stub"},
            },
        )
        self.addCleanup(app.stop)
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "recording_stub.py",
            make_recording_plugin_source("recording_stub", log_path),
        )
        app.start()

        c1 = app.post_json("/api/chat", {"session_id": "tog", "message": "T1"})
        self.assertEqual(c1.status, 200)
        self.assertIs(c1.data["t2i_hook"]["invoked"], False)
        self.assertEqual(len(read_jsonl(log_path)), 0)

        self.assertEqual(app.put_json("/api/config", {"t2i": {"enabled": True}}).status, 200)
        c2 = app.post_json("/api/chat", {"session_id": "tog", "message": "T2"})
        self.assertEqual(c2.status, 200)
        self.assertIs(c2.data["t2i_hook"]["invoked"], True)
        self.assertEqual(len(read_jsonl(log_path)), 1)

        self.assertEqual(app.put_json("/api/config", {"t2i": {"enabled": False}}).status, 200)
        c3 = app.post_json("/api/chat", {"session_id": "tog", "message": "T3"})
        self.assertEqual(c3.status, 200)
        self.assertIs(c3.data["t2i_hook"]["invoked"], False)
        self.assertEqual(len(read_jsonl(log_path)), 1)

    def test_f5_t1_06_enabled_with_noop_plugin_does_not_invoke_generation(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="noop回复")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "noop"},
            }
        )
        chat = app.post_json("/api/chat", {"session_id": "noop-s", "message": "测试noop"})
        self.assertEqual(chat.status, 200)
        hook = chat.data["t2i_hook"]
        self.assertEqual(hook["plugin_name"], "noop")
        self.assertIs(hook["invoked"], False)
        self.assertIn(hook["status"], ("skipped", "disabled"))
        self.assertFalse(hook.get("image_url"))
        self.assertFalse(hook.get("image_path"))


class TestTier1F6HttpApiAndWebUI(_Tier1Base):
    """F6: Lightweight HTTP API & Web UI (6 Tier 1 tests)."""

    def test_f6_t1_01_static_assets_and_offline_compliance(self) -> None:
        app = self._start_app()
        r_html = app.get_text("/")
        r_css = app.get_text("/static/style.css")
        r_js = app.get_text("/static/app.js")

        self.assertEqual(r_html.status, 200)
        self.assertEqual(r_css.status, 200)
        self.assertEqual(r_js.status, 200)
        self.assertIn("text/html", r_html.headers.get("content-type", ""))
        self.assertIn("text/css", r_css.headers.get("content-type", ""))
        self.assertIn("javascript", r_js.headers.get("content-type", ""))

        # Zero external CDN scripts/stylesheets
        cdn_pattern = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)
        self.assertIsNone(cdn_pattern.search(r_html.text))
        self.assertNotIn("cdn.", r_html.text.lower())
        self.assertNotIn("unpkg.com", r_html.text.lower())
        self.assertNotIn("jsdelivr", r_html.text.lower())

    def test_f6_t1_02_health_endpoint_contract(self) -> None:
        app = self._start_app()
        resp = app.get_json("/api/health")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.data["status"], "ok")
        self.assertIn("engine", resp.data)
        self.assertIn("skill", resp.data)
        self.assertIn("t2i", resp.data)

    def test_f6_t1_03_skill_and_extensions_status_endpoints(self) -> None:
        app = self._start_app()
        s_init = app.get_json("/api/skill/current")
        self.assertEqual(s_init.status, 200)
        self.assertIs(s_init.data.get("loaded", False), False)

        load_r = app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)})
        self.assertEqual(load_r.status, 200)

        s_after = app.get_json("/api/skill/current")
        self.assertEqual(s_after.status, 200)
        self.assertEqual(s_after.data["character_name"], "yae-miko-perspective")

        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        self.assertIn("enabled", ext.data)
        self.assertIn("active_plugin", ext.data)
        self.assertIsInstance(ext.data["plugins"], list)

    def test_f6_t1_04_chat_and_session_reset_endpoints(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="HTTP API 正常回复")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        app.post_json("/api/skill/load", {"path": str(YAE_MIKO_SKILL_FILE)})
        c1 = app.post_json("/api/chat", {"session_id": "api-1", "message": "第一句"})
        self.assertEqual(c1.status, 200)
        for key in ("session_id", "turn_index", "reply", "usage", "t2i_hook", "messages_count"):
            self.assertIn(key, c1.data)

        rst = app.post_json("/api/sessions/reset", {"session_id": "api-1"})
        self.assertEqual(rst.status, 200)
        self.assertEqual(app.get_json("/api/skill/current").data["character_name"], "yae-miko-perspective")

        c2 = app.post_json("/api/chat", {"session_id": "api-1", "message": "重置后"})
        self.assertEqual(c2.status, 200)
        self.assertEqual(c2.data["turn_index"], 1)

    def test_f6_t1_05_headless_edge_dom_verification(self) -> None:
        app = self._start_app()
        exe = find_browser_executable()
        self.assertIsNotNone(exe, "msedge.exe or chrome.exe must be available for headless DOM verification")
        rc, dom, stderr, _ = run_headless_edge_dom(app.base_url)
        self.assertEqual(rc, 0, f"Headless DOM dump failed: {stderr}")
        self.assertIn("<html", dom.lower())
        dom_lower = dom.lower()
        # Verify interactive UI controls exist in rendered DOM
        self.assertTrue("textarea" in dom_lower or "<input" in dom_lower)
        self.assertIn("button", dom_lower)
        self.assertTrue("skill" in dom_lower or "角色" in dom)
        self.assertTrue("t2i" in dom_lower or "文生图" in dom)

    def test_f6_t1_06_engine_lifecycle_endpoints(self) -> None:
        fake_llama = self._start_fake_llama()
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        s0 = app.get_json("/api/engine/status")
        self.assertEqual(s0.status, 200)
        s1 = app.post_json("/api/engine/start", {})
        self.assertEqual(s1.status, 200)
        self.assertEqual(s1.data["state"], "ready")
        s2 = app.post_json("/api/engine/stop", {})
        self.assertEqual(s2.status, 200)
        self.assertEqual(s2.data["state"], "stopped")


class TestTier1F7LauncherAndCli(_Tier1Base):
    """F7: Windows .bat Launcher & CLI (5 Tier 1 tests)."""

    def test_f7_t1_01_app_py_help_documents_cli_flags(self) -> None:
        app_py = require_target_file(REPO_ROOT / "app.py")
        proc = subprocess.run(
            [sys.executable, str(app_py), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        combined = proc.stdout + "\n" + proc.stderr
        for flag in ("--host", "--port", "--config", "--no-browser"):
            self.assertIn(flag, combined)

    def test_f7_t1_02_direct_cli_launch_and_config_isolation(self) -> None:
        tmp = self._make_temp_dir("moris_f7_02_")
        cfg = tmp / "cli_cfg.json"
        app = self._start_app(config_path=cfg, write_config=False)
        self.assertTrue(cfg.is_file())
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f7_t1_03_start_bat_forwards_args_and_serves_ui(self) -> None:
        app = self._start_app(use_bat_launcher=True)
        self.assertEqual(app.get_text("/").status, 200)
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f7_t1_04_start_bat_cwd_independence(self) -> None:
        outside_cwd = self._make_temp_dir("moris_f7_cwd_")
        app = self._start_app(use_bat_launcher=True, cwd=outside_cwd)
        home = app.get_text("/")
        self.assertEqual(home.status, 200)
        self.assertIn("<html", home.text.lower())
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f7_t1_05_clean_shutdown_of_managed_engine_subprocess(self) -> None:
        tmp = self._make_temp_dir("moris_f7_05_")
        exe = build_mock_llama_exe(tmp, "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        llama_port = find_free_port()

        app = AppProcessHarness(
            use_bat_launcher=False,
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": llama_port,
                    "startup_timeout_sec": 10,
                }
            },
        ).start()
        try:
            st = app.post_json("/api/engine/start", {})
            self.assertEqual(st.status, 200)
            pid = int(st.data["pid"])
            self.assertTrue(is_pid_alive(pid))
            app.send_ctrl_break()
            if app.proc is not None:
                try:
                    app.proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            app.stop()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and (is_pid_alive(pid) or is_port_open("127.0.0.1", llama_port)):
            time.sleep(0.1)
        self.assertFalse(is_pid_alive(pid))
        self.assertFalse(is_port_open("127.0.0.1", llama_port))


class TestTier1F8DocsAndExtensibility(_Tier1Base):
    """F8: Documentation & Extensibility Guide (5 Tier 1 tests)."""

    def test_f8_t1_01_docs_and_reference_files_exist_utf8(self) -> None:
        required = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "EXTENSIONS.md",
            REPO_ROOT / "config.example.json",
            REPO_ROOT / "plugins" / "t2i" / "example_stub.py",
        ]
        for path in required:
            require_target_file(path)
            content = path.read_text(encoding="utf-8")
            self.assertGreater(len(content.strip()), 20, f"{path} should not be empty")

    def test_f8_t1_02_readme_covers_quickstart_llama_skill_and_t2i(self) -> None:
        readme = require_target_file(REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("start.bat", readme)
        self.assertIn(".gguf", readme.lower())
        self.assertIn("llama", readme.lower())
        self.assertIn(".md", readme.lower())
        self.assertTrue("t2i" in readme.lower() or "文生图" in readme)

    def test_f8_t1_03_extensions_doc_covers_skill_and_t2i_contracts(self) -> None:
        doc = require_target_file(REPO_ROOT / "docs" / "EXTENSIONS.md").read_text(encoding="utf-8")
        for token in (
            ".md",
            "SKILL.md",
            "SceneContext",
            "T2IResult",
            "on_turn_completed",
            "PLUGIN",
            "register",
            "plugins/t2i",
        ):
            self.assertIn(token, doc)

    def test_f8_t1_04_doc_to_reality_cli_files_and_endpoints_consistency(self) -> None:
        readme = require_target_file(REPO_ROOT / "README.md").read_text(encoding="utf-8")
        ext_doc = require_target_file(REPO_ROOT / "docs" / "EXTENSIONS.md").read_text(encoding="utf-8")
        combined = readme + "\n" + ext_doc
        for rel in ("start.bat", "app.py", "config.example.json", "plugins/t2i/example_stub.py"):
            self.assertTrue((REPO_ROOT / rel).is_file())
            self.assertIn(rel.split("/")[-1], combined)

        help_proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "app.py"), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(help_proc.returncode, 0)
        for flag in ("--host", "--port", "--config", "--no-browser"):
            self.assertIn(flag, help_proc.stdout + help_proc.stderr)

    def test_f8_t1_05_live_t2i_extensibility_plugin_object_pattern(self) -> None:
        tmp = self._make_temp_dir("moris_f8_05_")
        log_path = tmp / "custom_t2i.jsonl"
        fake_llama = self._start_fake_llama(default_reply="扩展插件测试回复")

        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "custom_t2i"},
            },
        )
        self.addCleanup(app.stop)
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "custom_t2i.py",
            make_recording_plugin_source("custom_t2i", log_path, use_register=False),
        )
        app.start()

        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        self.assertIn("custom_t2i", {p["name"] for p in ext.data["plugins"]})

        chat = app.post_json("/api/chat", {"session_id": "f8-ext", "message": "场景测试"})
        self.assertEqual(chat.status, 200)
        self.assertEqual(chat.data["t2i_hook"]["plugin_name"], "custom_t2i")
        self.assertIs(chat.data["t2i_hook"]["invoked"], True)
        self.assertEqual(len(read_jsonl(log_path)), 1)


if __name__ == "__main__":
    unittest.main()
