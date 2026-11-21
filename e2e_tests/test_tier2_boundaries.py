"""Tier 2 E2E tests: Boundaries, error handling, and concurrency across F1-F8 (50 tests, zero skips)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from e2e_tests.harness import (
    AppProcessHarness,
    FIREFLY_SKILL_DIR,
    FakeLlamaServer,
    REPO_ROOT,
    build_mock_llama_exe,
    build_mock_python_exe,
    find_free_port,
    make_crashing_plugin_source,
    make_recording_plugin_source,
    make_single_md_skill,
    read_jsonl,
    require_target_file,
    write_dummy_gguf,
    write_t2i_plugin,
)


class _Tier2Base(unittest.TestCase):
    def _make_temp_dir(self, prefix: str = "moris_t2_") -> Path:
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

    def _assert_error_json(self, resp, expected_statuses: tuple[int, ...] = (400,)) -> dict:
        self.assertIn(resp.status, expected_statuses, f"Unexpected HTTP {resp.status}: {resp.text}")
        self.assertIsInstance(resp.data, dict, f"Expected JSON error, got: {resp.text}")
        self.assertIn("error", resp.data)
        err = resp.data["error"]
        self.assertIsInstance(err, dict)
        self.assertTrue(err.get("code"))
        self.assertTrue(err.get("message"))
        return err


class TestTier2F1Config(_Tier2Base):
    """F1: Atomic Config Store boundary tests (6 Tier 2 tests)."""

    def test_f1_t2_01_partial_preexisting_config_merges_defaults(self) -> None:
        tmp = self._make_temp_dir("moris_f1_t2_01_")
        cfg_path = tmp / "partial.json"
        port = find_free_port()
        cfg_path.write_text(
            json.dumps({"llama": {"temperature": 0.6}}),
            encoding="utf-8",
        )
        app = self._start_app(config_path=cfg_path, write_config=False, port=port)
        cfg = app.get_json("/api/config")
        self.assertEqual(cfg.status, 200)
        self.assertAlmostEqual(cfg.data["llama"]["temperature"], 0.6)
        self.assertIn("ctx_size", cfg.data["llama"])
        self.assertIn("server", cfg.data)
        self.assertIn("skill", cfg.data)
        self.assertIn("t2i", cfg.data)

    def test_f1_t2_02_invalid_values_rejected_atomically(self) -> None:
        tmp = self._make_temp_dir("moris_f1_t2_02_")
        cfg_path = tmp / "cfg.json"
        app = self._start_app(config_path=cfg_path)
        baseline_disk = cfg_path.read_text(encoding="utf-8")

        bad_patches = [
            {"server": {"port": 0}},
            {"server": {"port": 70000}},
            {"server": {"port": "abc"}},
            {"llama": {"ctx_size": -10}},
            {"llama": {"engine_mode": "invalid_mode"}},
        ]
        for patch in bad_patches:
            resp = app.put_json("/api/config", patch)
            self._assert_error_json(resp, (400,))
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), baseline_disk)

    def test_f1_t2_03_malformed_json_rejected_server_stays_healthy(self) -> None:
        tmp = self._make_temp_dir("moris_f1_t2_03_")
        cfg_path = tmp / "cfg.json"
        app = self._start_app(config_path=cfg_path)
        resp = app.request(
            "PUT",
            "/api/config",
            raw_body=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        self._assert_error_json(resp, (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)
        parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertIn("server", parsed)

    def test_f1_t2_04_concurrent_put_config_stress_no_corruption(self) -> None:
        tmp = self._make_temp_dir("moris_f1_t2_04_")
        cfg_path = tmp / "concurrent.json"
        app = self._start_app(config_path=cfg_path)

        def _worker(i: int) -> int:
            r = app.put_json("/api/config", {"llama": {"max_tokens": 200 + i}})
            return r.status

        with ThreadPoolExecutor(max_workers=10) as pool:
            statuses = list(pool.map(_worker, range(10)))

        self.assertEqual(statuses, [200] * 10)
        parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.assertIn(parsed["llama"]["max_tokens"], range(200, 210))
        leftovers = [p.name for p in cfg_path.parent.iterdir() if p.name != "concurrent.json"]
        self.assertEqual(leftovers, [])

    def test_f1_t2_05_boundary_values_accepted(self) -> None:
        app = self._start_app()
        resp = app.put_json(
            "/api/config",
            {
                "skill": {"max_prompt_chars": 0},
                "llama": {"n_gpu_layers": -1, "temperature": 0.0, "port": 65535},
                "t2i": {"context_window_turns": 1},
            },
        )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.data["skill"]["max_prompt_chars"], 0)
        self.assertEqual(resp.data["llama"]["n_gpu_layers"], -1)
        self.assertEqual(resp.data["llama"]["temperature"], 0.0)
        self.assertEqual(resp.data["llama"]["port"], 65535)
        self.assertEqual(resp.data["t2i"]["context_window_turns"], 1)

    def test_f1_t2_06_empty_object_and_unknown_extra_keys_safe(self) -> None:
        app = self._start_app()
        r_empty = app.put_json("/api/config", {})
        self.assertEqual(r_empty.status, 200)
        for sec in ("server", "skill", "llama", "t2i"):
            self.assertIn(sec, r_empty.data)

        r_extra = app.put_json("/api/config", {"unknown_top": {"x": 1}})
        self.assertIn(r_extra.status, (200, 400))
        self.assertEqual(app.get_json("/api/health").status, 200)


class TestTier2F2SkillLoader(_Tier2Base):
    """F2: .md-Only Skill Loader boundary tests (7 Tier 2 tests)."""

    def test_f2_t2_01_nonexistent_path_returns_400_no_crash(self) -> None:
        app = self._start_app()
        resp = app.post_json("/api/skill/load", {"path": r"C:\nonexistent_moris_path\missing_skill.md"})
        self._assert_error_json(resp, (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f2_t2_02_single_non_md_file_rejected(self) -> None:
        app = self._start_app()
        for non_md in (FIREFLY_SKILL_DIR / "manifest.json", FIREFLY_SKILL_DIR / "LICENSE"):
            resp = app.post_json("/api/skill/load", {"path": str(non_md)})
            self._assert_error_json(resp, (400,))
        curr = app.get_json("/api/skill/current")
        self.assertEqual(curr.status, 200)
        self.assertIs(curr.data.get("loaded", False), False)

    def test_f2_t2_03_empty_dir_and_non_md_only_dir_rejected(self) -> None:
        tmp = self._make_temp_dir("moris_f2_t2_03_")
        empty_dir = tmp / "empty_dir"
        empty_dir.mkdir()
        non_md_dir = tmp / "non_md_dir"
        non_md_dir.mkdir()
        (non_md_dir / "a.txt").write_text("hi", encoding="utf-8")
        (non_md_dir / "b.json").write_text("{}", encoding="utf-8")
        (non_md_dir / ".git").mkdir()
        (non_md_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

        app = self._start_app()
        self._assert_error_json(app.post_json("/api/skill/load", {"path": str(empty_dir)}), (400,))
        self._assert_error_json(app.post_json("/api/skill/load", {"path": str(non_md_dir)}), (400,))

    def test_f2_t2_04_whitespace_only_md_and_no_trailing_newline_concat(self) -> None:
        tmp = self._make_temp_dir("moris_f2_t2_04_")
        s_dir = tmp / "concat_skill"
        s_dir.mkdir()
        (s_dir / "SKILL.md").write_bytes(b"LastLineNoNewline")
        (s_dir / "empty.md").write_text("   \n\t\r\n  ", encoding="utf-8")
        (s_dir / "profile.md").write_text("# ProfileHeading", encoding="utf-8")

        app = self._start_app()
        resp = app.post_json("/api/skill/load", {"path": str(s_dir)})
        self.assertEqual(resp.status, 200)
        prompt = resp.data["system_prompt"]
        self.assertIn("LastLineNoNewline", prompt)
        self.assertIn("# ProfileHeading", prompt)
        self.assertNotIn("LastLineNoNewline# ProfileHeading", prompt)

    def test_f2_t2_05_max_prompt_chars_budget_and_warnings(self) -> None:
        app = self._start_app(config_overrides={"skill": {"max_prompt_chars": 3000}})
        resp = app.post_json(
            "/api/skill/load",
            {"path": str(FIREFLY_SKILL_DIR), "exclude_meta_docs": False},
        )
        self.assertEqual(resp.status, 200)
        data = resp.data
        self.assertTrue(data["warnings"], "Expected non-empty warnings when exceeding max_prompt_chars")
        self.assertGreater(data["estimated_tokens"], 0)
        self.assertLessEqual(data["total_chars"], 3000)
        self.assertIn("绝对不可违背的最高规则", data["system_prompt"])

    def test_f2_t2_06_invalid_utf8_encoding_graceful_error(self) -> None:
        tmp = self._make_temp_dir("moris_f2_t2_06_")
        bad_md = tmp / "bad_bytes.md"
        bad_md.write_bytes(b"\xff\xfe\x80\x81\xfa\xfb")

        app = self._start_app()
        resp = app.post_json("/api/skill/load", {"path": str(bad_md)})
        if resp.status != 200:
            self._assert_error_json(resp, (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f2_t2_07_empty_body_or_missing_path_field(self) -> None:
        app = self._start_app()
        self._assert_error_json(app.post_json("/api/skill/load", {}), (400,))
        self._assert_error_json(app.post_json("/api/skill/load", {"path": ""}), (400,))
        raw_bad = app.request(
            "POST",
            "/api/skill/load",
            raw_body=b"{not-json",
            headers={"Content-Type": "application/json"},
        )
        self._assert_error_json(raw_bad, (400,))


class TestTier2F3LlamaEngine(_Tier2Base):
    """F3: llama.cpp Engine Adapter error-matrix tests (8 Tier 2 tests)."""

    def test_f3_t2_01_config_incomplete_400(self) -> None:
        app = self._start_app(
            config_overrides={"llama": {"engine_mode": "managed_server", "llama_cpp_path": "", "model_path": ""}}
        )
        err1 = self._assert_error_json(app.post_json("/api/engine/start", {}), (400,))
        self.assertEqual(err1["code"], "CONFIG_INCOMPLETE")

        app.put_json("/api/config", {"llama": {"engine_mode": "external_server", "server_url": ""}})
        err2 = self._assert_error_json(app.post_json("/api/engine/start", {}), (400,))
        self.assertEqual(err2["code"], "CONFIG_INCOMPLETE")

    def test_f3_t2_02_engine_binary_not_found_400(self) -> None:
        tmp = self._make_temp_dir("moris_f3_t2_02_")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        empty_bin_dir = tmp / "empty_bin"
        empty_bin_dir.mkdir()

        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(tmp / "nonexistent_llama_server.exe"),
                    "model_path": str(gguf),
                }
            }
        )
        err1 = self._assert_error_json(app.post_json("/api/engine/start", {}), (400,))
        self.assertEqual(err1["code"], "ENGINE_BINARY_NOT_FOUND")

        app.put_json("/api/config", {"llama": {"llama_cpp_path": str(empty_bin_dir)}})
        err2 = self._assert_error_json(app.post_json("/api/engine/start", {}), (400,))
        self.assertEqual(err2["code"], "ENGINE_BINARY_NOT_FOUND")
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f3_t2_03_model_file_invalid_400(self) -> None:
        tmp = self._make_temp_dir("moris_f3_t2_03_")
        exe = build_mock_llama_exe(tmp / "bin", "llama-server.exe")
        model_dir = tmp / "model_dir"
        model_dir.mkdir()
        wrong_ext = tmp / "not_a_model.txt"
        wrong_ext.write_text("hello", encoding="utf-8")
        zero_gguf = tmp / "empty.gguf"
        zero_gguf.write_bytes(b"")

        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "managed_server", "llama_cpp_path": str(exe)}
            }
        )
        for bad_model in (
            tmp / "missing.gguf",
            model_dir,
            wrong_ext,
            zero_gguf,
        ):
            app.put_json("/api/config", {"llama": {"model_path": str(bad_model)}})
            err = self._assert_error_json(app.post_json("/api/engine/start", {}), (400,))
            self.assertEqual(err["code"], "MODEL_FILE_INVALID", f"Failed on {bad_model}")

    def test_f3_t2_04_engine_port_conflict_409(self) -> None:
        tmp = self._make_temp_dir("moris_f3_t2_04_")
        exe = build_mock_llama_exe(tmp / "bin", "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        busy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy_sock.bind(("127.0.0.1", 0))
        busy_sock.listen(1)
        busy_port = busy_sock.getsockname()[1]
        self.addCleanup(busy_sock.close)

        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": busy_port,
                }
            }
        )
        err = self._assert_error_json(app.post_json("/api/engine/start", {}), (409,))
        self.assertEqual(err["code"], "ENGINE_PORT_CONFLICT")

    def test_f3_t2_05_engine_process_exited_502(self) -> None:
        tmp = self._make_temp_dir("moris_f3_t2_05_")
        exe = build_mock_llama_exe(tmp / "bin", "llama-server.exe")
        gguf = write_dummy_gguf(tmp / "tiny.gguf")
        app = self._start_app(
            env_overrides={"MOCK_LLAMA_MODE": "crash"},
            config_overrides={
                "llama": {
                    "engine_mode": "managed_server",
                    "llama_cpp_path": str(exe),
                    "model_path": str(gguf),
                    "port": find_free_port(),
                    "startup_timeout_sec": 5,
                }
            },
        )
        err = self._assert_error_json(app.post_json("/api/engine/start", {}), (502,))
        self.assertEqual(err["code"], "ENGINE_PROCESS_EXITED")
        st = app.get_json("/api/engine/status")
        self.assertEqual(st.status, 200)
        self.assertEqual(st.data["state"], "error")

    def test_f3_t2_06_engine_loading_503(self) -> None:
        fake_llama = self._start_fake_llama(health_status=503)
        fake_llama.set_mode("http_503")
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "startup_timeout_sec": 2,
                }
            }
        )
        err = self._assert_error_json(app.post_json("/api/engine/start", {}), (503,))
        self.assertEqual(err["code"], "ENGINE_LOADING")

    def test_f3_t2_07_engine_timeout_504(self) -> None:
        fake_llama = self._start_fake_llama(response_delay_sec=3.5)
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "request_timeout_sec": 1,
                }
            }
        )
        t0 = time.monotonic()
        resp = app.post_json("/api/chat", {"session_id": "to-1", "message": "慢响应"})
        elapsed = time.monotonic() - t0
        err = self._assert_error_json(resp, (504,))
        self.assertEqual(err["code"], "ENGINE_TIMEOUT")
        self.assertLess(elapsed, 4.0)

    def test_f3_t2_08_engine_upstream_error_502(self) -> None:
        dead_port = find_free_port()
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": f"http://127.0.0.1:{dead_port}",
                }
            }
        )
        err_dead = self._assert_error_json(app.post_json("/api/chat", {"session_id": "u1", "message": "hi"}), (502,))
        self.assertEqual(err_dead["code"], "ENGINE_UPSTREAM_ERROR")

        fake_llama = self._start_fake_llama(default_reply="恢复成功")
        app.put_json("/api/config", {"llama": {"server_url": fake_llama.base_url}})

        fake_llama.enqueue_response(mode="http_500")
        err_500 = self._assert_error_json(app.post_json("/api/chat", {"session_id": "u2", "message": "hi"}), (502,))
        self.assertEqual(err_500["code"], "ENGINE_UPSTREAM_ERROR")

        fake_llama.enqueue_response(mode="malformed_json")
        err_bad = self._assert_error_json(app.post_json("/api/chat", {"session_id": "u2", "message": "hi"}), (502,))
        self.assertEqual(err_bad["code"], "ENGINE_UPSTREAM_ERROR")

        ok = app.post_json("/api/chat", {"session_id": "u2", "message": "recovery"})
        self.assertEqual(ok.status, 200)
        self.assertEqual(ok.data["reply"], "恢复成功")


class TestTier2F4MultiTurnChat(_Tier2Base):
    """F4: Multi-Turn Chat Service boundary tests (7 Tier 2 tests)."""

    def test_f4_t2_01_empty_and_whitespace_message_rejected_no_history_pollution(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"ok-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        self.assertEqual(app.post_json("/api/chat", {"session_id": "s-emp", "message": "Turn1"}).status, 200)

        for bad in ({"session_id": "s-emp", "message": ""}, {"session_id": "s-emp", "message": "   \n\t "}, {"session_id": "s-emp"}):
            self._assert_error_json(app.post_json("/api/chat", bad), (400,))

        self.assertEqual(len(fake_llama.requests), 1)
        t2 = app.post_json("/api/chat", {"session_id": "s-emp", "message": "Turn2"})
        self.assertEqual(t2.status, 200)
        self.assertEqual(t2.data["turn_index"], 2)

    def test_f4_t2_02_failed_turn_does_not_corrupt_session_history(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"good-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        self.assertEqual(app.post_json("/api/chat", {"session_id": "rb", "message": "U1"}).status, 200)

        fake_llama.enqueue_response(mode="http_500")
        fail = app.post_json("/api/chat", {"session_id": "rb", "message": "FAILED_MSG"})
        self._assert_error_json(fail, (502,))

        retry = app.post_json("/api/chat", {"session_id": "rb", "message": "RETRY_MSG"})
        self.assertEqual(retry.status, 200)
        self.assertEqual(retry.data["turn_index"], 2)
        sent_dump = json.dumps(fake_llama.last_request["body"]["messages"], ensure_ascii=False)
        self.assertNotIn("FAILED_MSG", sent_dump)
        self.assertIn("RETRY_MSG", sent_dump)

    def test_f4_t2_03_chat_without_loaded_skill(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"noskill-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        r1 = app.post_json("/api/chat", {"session_id": "no-sk", "message": "First"})
        r2 = app.post_json("/api/chat", {"session_id": "no-sk", "message": "Second"})
        self.assertEqual(r1.status, 200)
        self.assertEqual(r2.status, 200)
        msgs = fake_llama.last_request["body"]["messages"]
        contents = [m["content"] for m in msgs]
        self.assertIn("First", contents)
        self.assertIn("Second", contents)

    def test_f4_t2_04_omitted_session_id_and_reset_nonexistent_session(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="auto-id")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        c1 = app.post_json("/api/chat", {"message": "hello"})
        self.assertEqual(c1.status, 200)
        sid = c1.data["session_id"]
        self.assertTrue(isinstance(sid, str) and len(sid) > 0)

        c2 = app.post_json("/api/chat", {"session_id": sid, "message": "follow up"})
        self.assertEqual(c2.status, 200)
        self.assertEqual(c2.data["turn_index"], 2)

        rst = app.post_json("/api/sessions/reset", {"session_id": "never-existed-999"})
        self.assertEqual(rst.status, 200)

    def test_f4_t2_05_max_history_turns_one_and_zero_boundary(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"ans-{idx}")
        app = self._start_app(
            config_overrides={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                    "max_history_turns": 1,
                }
            }
        )
        tmp = self._make_temp_dir("moris_f4_t2_05_")
        s_path = make_single_md_skill(tmp, "sys.md", "PINNED_SYS", {"name": "Pin"})
        app.post_json("/api/skill/load", {"path": str(s_path)})

        for k in (1, 2, 3):
            self.assertEqual(app.post_json("/api/chat", {"session_id": "w1", "message": f"M{k}"}).status, 200)

        msgs = fake_llama.last_request["body"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual([(m["role"], m["content"]) for m in msgs[1:]], [("user", "M2"), ("assistant", "ans-2"), ("user", "M3")])

        # Zero boundary: either rejected as 400 validation or keeps 0 prior turns cleanly
        z_resp = app.put_json("/api/config", {"llama": {"max_history_turns": 0}})
        self.assertIn(z_resp.status, (200, 400))

    def test_f4_t2_06_unicode_emoji_multiline_and_large_user_message(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="收到长消息✨")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        large_msg = ("星穹铁道🔥✨\n\"引号\"与\\反斜杠\n" + ("测试文本" * 1950))
        resp = app.post_json("/api/chat", {"session_id": "uni", "message": large_msg})
        self.assertEqual(resp.status, 200)
        sent_msg = fake_llama.last_request["body"]["messages"][-1]["content"]
        self.assertEqual(sent_msg, large_msg)

    def test_f4_t2_07_concurrent_chats_across_sessions(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"echo:{body['messages'][-1]['content']}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )

        def _send(i: int):
            return app.post_json("/api/chat", {"session_id": f"conc-{i}", "message": f"msg-{i}"})

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_send, range(4)))

        for i, r in enumerate(results):
            self.assertEqual(r.status, 200)
            self.assertEqual(r.data["session_id"], f"conc-{i}")
            self.assertEqual(r.data["reply"], f"echo:msg-{i}")


class TestTier2F5T2IExtension(_Tier2Base):
    """F5: Post-Turn T2I Extension Interface boundary tests (6 Tier 2 tests)."""

    def test_f5_t2_01_faulty_plugin_exception_isolated_from_chat(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="对话不受插件崩溃影响")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "crashing_stub"},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "crashing_stub.py",
            make_crashing_plugin_source("crashing_stub", "GPU OOM simulated"),
        )
        app.start()
        self.addCleanup(app.stop)

        for turn in (1, 2):
            c = app.post_json("/api/chat", {"session_id": "crash-t2i", "message": f"hi {turn}"})
            self.assertEqual(c.status, 200)
            self.assertEqual(c.data["reply"], "对话不受插件崩溃影响")
            self.assertEqual(c.data["t2i_hook"]["status"], "error")
            self.assertIn("GPU OOM simulated", c.data["t2i_hook"]["message"])
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f5_t2_02_unknown_active_plugin_no_crash(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="未知插件回退")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "missing_plugin_xyz"},
            }
        )
        self.assertEqual(app.get_json("/api/health").status, 200)
        c = app.post_json("/api/chat", {"session_id": "unk-p", "message": "hi"})
        self.assertEqual(c.status, 200)
        self.assertEqual(c.data["reply"], "未知插件回退")
        self.assertEqual(c.data["t2i_hook"]["status"], "error")
        self.assertIs(c.data["t2i_hook"]["invoked"], False)

    def test_f5_t2_03_syntax_error_plugin_file_isolated_during_discovery(self) -> None:
        tmp = self._make_temp_dir("moris_f5_t2_03_")
        log_path = tmp / "good.jsonl"
        fake_llama = self._start_fake_llama(default_reply="好插件可用")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "good_stub"},
            },
        )
        p_dir = app.temp_dir / "plugins" / "t2i"
        write_t2i_plugin(p_dir, "broken_syntax.py", "def broken(:\n    pass\n")
        write_t2i_plugin(p_dir, "good_stub.py", make_recording_plugin_source("good_stub", log_path))
        app.start()
        self.addCleanup(app.stop)

        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        names = {p["name"] for p in ext.data["plugins"]}
        self.assertIn("noop", names)
        self.assertIn("good_stub", names)

        c = app.post_json("/api/chat", {"session_id": "syn", "message": "test"})
        self.assertEqual(c.status, 200)
        self.assertEqual(c.data["t2i_hook"]["status"], "stub_ok")
        self.assertEqual(len(read_jsonl(log_path)), 1)

    def test_f5_t2_04_non_plugin_files_and_underscored_helpers_ignored(self) -> None:
        fake_llama = self._start_fake_llama()
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
            },
        )
        p_dir = app.temp_dir / "plugins" / "t2i"
        write_t2i_plugin(p_dir, "__init__.py", "")
        write_t2i_plugin(p_dir, "_private_helper.py", "HELPER_VAL = 1\n")
        write_t2i_plugin(p_dir, "notes.txt", "not python")
        write_t2i_plugin(p_dir, "no_entrypoint.py", "X = 42\n")
        app.start()
        self.addCleanup(app.stop)

        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        names = {p["name"] for p in ext.data["plugins"]}
        for ignored in ("__init__", "_private_helper", "notes", "no_entrypoint"):
            self.assertNotIn(ignored, names)

    def test_f5_t2_05_hook_not_invoked_when_llm_turn_fails(self) -> None:
        tmp = self._make_temp_dir("moris_f5_t2_05_")
        log_path = tmp / "never_invoked.jsonl"
        fake_llama = self._start_fake_llama()
        fake_llama.set_mode("http_500")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "recording_stub"},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "recording_stub.py",
            make_recording_plugin_source("recording_stub", log_path),
        )
        app.start()
        self.addCleanup(app.stop)

        resp = app.post_json("/api/chat", {"session_id": "fail-llm", "message": "boom"})
        self._assert_error_json(resp, (502,))
        self.assertEqual(len(read_jsonl(log_path)), 0)

    def test_f5_t2_06_zero_context_window_and_missing_plugins_dir(self) -> None:
        tmp = self._make_temp_dir("moris_f5_t2_06_")
        fake_llama = self._start_fake_llama(default_reply="ok")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {
                    "enabled": False,
                    "context_window_turns": 1,
                    "plugins_dir": str(tmp / "nonexistent_plugins_dir"),
                },
            }
        )
        ext = app.get_json("/api/extensions/t2i")
        self.assertEqual(ext.status, 200)
        self.assertIn("noop", {p["name"] for p in ext.data["plugins"]})
        self.assertEqual(app.post_json("/api/chat", {"session_id": "z", "message": "hi"}).status, 200)

        bad_zero = app.put_json("/api/config", {"t2i": {"context_window_turns": 0}})
        self._assert_error_json(bad_zero, (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)


class TestTier2F6HttpApiAndWebUI(_Tier2Base):
    """F6: Lightweight HTTP API & Web UI boundary tests (6 Tier 2 tests)."""

    def test_f6_t2_01_malformed_json_across_post_and_put_endpoints(self) -> None:
        app = self._start_app()
        for method, path in (
            ("POST", "/api/chat"),
            ("POST", "/api/skill/load"),
            ("POST", "/api/sessions/reset"),
            ("PUT", "/api/config"),
        ):
            r = app.request(
                method,
                path,
                raw_body=b"{malformed",
                headers={"Content-Type": "application/json"},
            )
            self._assert_error_json(r, (400,))

    def test_f6_t2_02_unknown_api_route_and_missing_static_404(self) -> None:
        app = self._start_app()
        self._assert_error_json(app.get_json("/api/nonexistent"), (404,))
        self._assert_error_json(app.post_json("/api/nonexistent", {}), (404,))
        missing_static = app.get_text("/static/nonexistent_asset_xyz.js")
        self.assertEqual(missing_static.status, 404)

    def test_f6_t2_03_path_traversal_protection_on_static(self) -> None:
        app = self._start_app()
        for path in ("/static/../../app.py", "/static/%2e%2e/config.example.json", "/static/../app.py"):
            r = app.get_text(path)
            self.assertIn(r.status, (400, 403, 404))
            self.assertNotIn("ThreadingHTTPServer", r.text)
            self.assertNotIn("engine_mode", r.text)

    def test_f6_t2_04_invalid_semantic_payloads_400(self) -> None:
        app = self._start_app()
        self._assert_error_json(app.post_json("/api/chat", {"message": "   "}), (400,))
        self._assert_error_json(app.post_json("/api/chat", {}), (400,))
        self._assert_error_json(app.post_json("/api/skill/load", {}), (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f6_t2_05_empty_body_and_zero_content_length_400(self) -> None:
        app = self._start_app()
        for method, path in (("POST", "/api/chat"), ("PUT", "/api/config")):
            r = app.request(
                method,
                path,
                raw_body=b"",
                headers={"Content-Type": "application/json", "Content-Length": "0"},
            )
            self._assert_error_json(r, (400,))
        self.assertEqual(app.get_json("/api/health").status, 200)

    def test_f6_t2_06_concurrent_api_calls_across_threads(self) -> None:
        fake_llama = self._start_fake_llama()
        fake_llama.set_reply_fn(lambda body, idx: f"r-{body['messages'][-1]['content']}")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )

        def _task(i: int) -> tuple[int, int, int]:
            h = app.get_json("/api/health")
            c = app.get_json("/api/config")
            ch = app.post_json("/api/chat", {"session_id": f"th-{i}", "message": f"p-{i}"})
            return h.status, c.status, ch.status

        with ThreadPoolExecutor(max_workers=6) as pool:
            res = list(pool.map(_task, range(6)))

        self.assertEqual(res, [(200, 200, 200)] * 6)


class TestTier2F7LauncherAndCli(_Tier2Base):
    """F7: Windows .bat Launcher & CLI boundary tests (5 Tier 2 tests)."""

    def test_f7_t2_01_port_conflict_exits_nonzero_with_error(self) -> None:
        app_py = require_target_file(REPO_ROOT / "app.py")
        tmp = self._make_temp_dir("moris_f7_t2_01_")
        cfg = tmp / "cfg.json"
        busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        busy_port = busy.getsockname()[1]
        self.addCleanup(busy.close)

        proc = subprocess.run(
            [
                sys.executable,
                str(app_py),
                "--host",
                "127.0.0.1",
                "--port",
                str(busy_port),
                "--config",
                str(cfg),
                "--no-browser",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue((proc.stdout + proc.stderr).strip())

    def test_f7_t2_02_missing_or_unsupported_python_in_start_bat(self) -> None:
        start_bat = require_target_file(REPO_ROOT / "start.bat")
        tmp = self._make_temp_dir("moris_f7_t2_02_")
        fake_bin = tmp / "fake_bin"
        fake_bin.mkdir()
        # Mock unsupported python 3.9 and ensure py launcher is shadowed by compiled .exe stubs
        build_mock_python_exe(fake_bin, "python.exe", "Python 3.9.0", 1)
        build_mock_python_exe(fake_bin, "py.exe", "", 9009)

        env = os.environ.copy()
        env["PATH"] = f"{fake_bin};C:\\Windows\\System32"

        proc = subprocess.run(
            ["cmd.exe", "/c", str(start_bat), "--no-browser"],
            cwd=str(REPO_ROOT),
            input="\r\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=8,
        )
        self.assertNotEqual(proc.returncode, 0)
        combined = (proc.stdout + "\n" + proc.stderr).lower()
        self.assertTrue("python" in combined or "3.11" in combined)

    def test_f7_t2_03_spaces_and_unicode_in_config_path_via_start_bat(self) -> None:
        tmp = self._make_temp_dir("moris_f7_t2_03_")
        uni_dir = tmp / "角色 配置"
        uni_dir.mkdir()
        cfg_path = uni_dir / "my config.json"
        app = self._start_app(use_bat_launcher=True, config_path=cfg_path, write_config=False)
        self.assertEqual(app.get_json("/api/health").status, 200)
        self.assertTrue(cfg_path.is_file())
        put = app.put_json("/api/config", {"skill": {"last_path": "测试路径"}})
        self.assertEqual(put.status, 200)
        self.assertEqual(json.loads(cfg_path.read_text(encoding="utf-8"))["skill"]["last_path"], "测试路径")

    def test_f7_t2_04_invalid_cli_arguments_exit_nonzero(self) -> None:
        app_py = require_target_file(REPO_ROOT / "app.py")
        for bad_args in (["--bogus-arg"], ["--port", "99999"]):
            proc = subprocess.run(
                [sys.executable, str(app_py), *bad_args],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=6,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue((proc.stdout + proc.stderr).strip())

    def test_f7_t2_05_open_browser_false_config_without_no_browser_flag(self) -> None:
        app = self._start_app(
            omit_no_browser=True,
            config_overrides={"server": {"open_browser": False}},
        )
        self.assertEqual(app.get_json("/api/health").status, 200)
        cfg = app.get_json("/api/config")
        self.assertIs(cfg.data["server"]["open_browser"], False)


class TestTier2F8DocsAndExtensibility(_Tier2Base):
    """F8: Documentation & Extensibility boundary tests (5 Tier 2 tests)."""

    def test_f8_t2_01_live_t2i_extensibility_register_pattern(self) -> None:
        tmp = self._make_temp_dir("moris_f8_t2_01_")
        log_path = tmp / "reg_ext.jsonl"
        fake_llama = self._start_fake_llama(default_reply="register扩展回复")
        app = AppProcessHarness(
            isolated_plugins=True,
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "ext_register_plugin"},
            },
        )
        write_t2i_plugin(
            app.temp_dir / "plugins" / "t2i",
            "ext_register_plugin.py",
            make_recording_plugin_source("ext_register_plugin", log_path, use_register=True),
        )
        app.start()
        self.addCleanup(app.stop)

        c = app.post_json("/api/chat", {"session_id": "f8-reg", "message": "测试register扩展"})
        self.assertEqual(c.status, 200)
        self.assertEqual(c.data["t2i_hook"]["plugin_name"], "ext_register_plugin")
        self.assertIs(c.data["t2i_hook"]["invoked"], True)
        self.assertEqual(len(read_jsonl(log_path)), 1)

    def test_f8_t2_02_live_skill_extensibility_custom_folder(self) -> None:
        tmp = self._make_temp_dir("moris_f8_t2_02_")
        custom_dir = tmp / "new_character_skill"
        custom_dir.mkdir()
        (custom_dir / "SKILL.md").write_text("---\nname: CustomStar\n---\n# Star Rule", encoding="utf-8")
        (custom_dir / "personality.md").write_text("Cheerful and brave.", encoding="utf-8")
        (custom_dir / "noise.yaml").write_text("SHOULD_NOT_BE_READ: true", encoding="utf-8")

        fake_llama = self._start_fake_llama(default_reply="新角色回复")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url}
            }
        )
        load = app.post_json("/api/skill/load", {"path": str(custom_dir)})
        self.assertEqual(load.status, 200)
        self.assertEqual(load.data["character_name"], "CustomStar")
        self.assertIn("noise.yaml", load.data["ignored_files"])
        self.assertNotIn("SHOULD_NOT_BE_READ", load.data["system_prompt"])

        chat = app.post_json("/api/chat", {"session_id": "new-char", "message": "Hi"})
        self.assertEqual(chat.status, 200)
        sys_msg = fake_llama.last_request["body"]["messages"][0]["content"]
        self.assertIn("Cheerful and brave.", sys_msg)

    def test_f8_t2_03_no_broken_local_file_references_in_docs(self) -> None:
        readme = require_target_file(REPO_ROOT / "README.md").read_text(encoding="utf-8")
        ext_doc = require_target_file(REPO_ROOT / "docs" / "EXTENSIONS.md").read_text(encoding="utf-8")
        known_files = (
            "start.bat",
            "app.py",
            "config.example.json",
            "backend/config_store.py",
            "backend/skill_loader.py",
            "backend/llama_engine.py",
            "backend/chat_service.py",
            "backend/server.py",
            "backend/extensions/t2i.py",
            "plugins/t2i/example_stub.py",
            "static/index.html",
            "static/style.css",
            "static/app.js",
            "docs/EXTENSIONS.md",
        )
        combined = readme + "\n" + ext_doc
        for rel in known_files:
            if rel in combined:
                self.assertTrue((REPO_ROOT / rel).is_file(), f"Doc references missing file: {rel}")

    def test_f8_t2_04_builtin_example_stub_runs_with_no_external_network(self) -> None:
        fake_llama = self._start_fake_llama(default_reply="参考插件回复")
        app = self._start_app(
            config_overrides={
                "llama": {"engine_mode": "external_server", "server_url": fake_llama.base_url},
                "t2i": {"enabled": True, "active_plugin": "example_stub"},
            }
        )
        chat = app.post_json("/api/chat", {"session_id": "ex-stub", "message": "星海"})
        self.assertEqual(chat.status, 200)
        hook = chat.data["t2i_hook"]
        self.assertIs(hook["invoked"], True)
        self.assertEqual(hook["plugin_name"], "example_stub")
        self.assertNotEqual(hook["status"], "error")

    def test_f8_t2_05_zero_third_party_deps_python_isolated_mode(self) -> None:
        app_py = require_target_file(REPO_ROOT / "app.py")
        self.assertFalse((REPO_ROOT / "package.json").exists())
        self.assertFalse((REPO_ROOT / "node_modules").exists())
        proc = subprocess.run(
            [sys.executable, "-I", str(app_py), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--port", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
