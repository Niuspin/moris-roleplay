"""Unit tests for backend.config_store and config.example.json."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from backend.config_store import DEFAULT_CONFIG, ConfigError, ConfigStore

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigStoreTest(unittest.TestCase):
    def test_config_example_matches_default_config(self) -> None:
        example_path = REPO_ROOT / "config.example.json"
        self.assertTrue(example_path.is_file(), "config.example.json must exist at repo root")
        parsed = json.loads(example_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, DEFAULT_CONFIG)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(Path(tmpdir) / "config.json")
            self.assertEqual(store.load(), parsed)

    def test_load_creates_default_config_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "nested" / "dir" / "config.json"
            self.assertFalse(cfg_path.exists())

            store = ConfigStore(cfg_path)
            loaded = store.load()
            self.assertEqual(loaded, DEFAULT_CONFIG)
            self.assertTrue(cfg_path.is_file())
            self.assertEqual(list(cfg_path.parent.glob("*.tmp")), [])

            # Mutating returned dict must not affect DEFAULT_CONFIG or subsequent loads
            loaded["server"]["port"] = 1234
            loaded["t2i"]["plugin_settings"]["injected"] = True
            self.assertEqual(store.load(), DEFAULT_CONFIG)

    def test_load_uses_optional_example_path_without_mutating_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            example_path = Path(tmpdir) / "config.example.json"
            custom_example = copy.deepcopy(DEFAULT_CONFIG)
            custom_example["server"]["port"] = 9999
            custom_example["llama"]["ctx_size"] = 8192
            original_text = json.dumps(custom_example, indent=2)
            example_path.write_text(original_text, encoding="utf-8")

            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path, example_path=example_path)
            loaded = store.load()

            self.assertEqual(loaded["server"]["port"], 9999)
            self.assertEqual(loaded["llama"]["ctx_size"], 8192)
            self.assertEqual(example_path.read_text(encoding="utf-8"), original_text)

    def test_load_deep_merges_partial_config_preserving_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            partial = {"server": {"port": 9001}, "llama": {"temperature": 0.55}}
            partial_text = json.dumps(partial)
            cfg_path.write_text(partial_text, encoding="utf-8")

            store = ConfigStore(cfg_path)
            loaded = store.load()

            self.assertEqual(loaded["server"]["port"], 9001)
            self.assertEqual(loaded["server"]["host"], DEFAULT_CONFIG["server"]["host"])
            self.assertEqual(loaded["llama"]["temperature"], 0.55)
            self.assertEqual(loaded["llama"]["ctx_size"], DEFAULT_CONFIG["llama"]["ctx_size"])
            # Partial file on disk should not have been overwritten by load()
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), partial_text)

    def test_load_handles_utf8_bom_and_unicode_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            chinese_skill_path = r"C:\Users\nzp18\Desktop\角色skill\流萤\firefly-skill"
            payload = copy.deepcopy(DEFAULT_CONFIG)
            payload["skill"]["last_path"] = chinese_skill_path
            cfg_path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8-sig"))

            store = ConfigStore(cfg_path)
            loaded = store.load()
            self.assertEqual(loaded["skill"]["last_path"], chinese_skill_path)

            # Also verify update + reload roundtrip with CJK path
            store2 = ConfigStore(Path(tmpdir) / "roundtrip.json")
            store2.update({"skill": {"last_path": chinese_skill_path}})
            reloaded = ConfigStore(Path(tmpdir) / "roundtrip.json").load()
            self.assertEqual(reloaded["skill"]["last_path"], chinese_skill_path)

    def test_load_corrupt_json_and_empty_file_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [("corrupt.json", "{broken"), ("empty.json", "")]:
                with self.subTest(name=name):
                    cfg_path = Path(tmpdir) / name
                    cfg_path.write_text(content, encoding="utf-8")
                    store = ConfigStore(cfg_path)
                    with self.assertRaises(ConfigError) as ctx:
                        store.load()
                    self.assertEqual(ctx.exception.code, "CONFIG_INVALID_JSON")
                    self.assertEqual(cfg_path.read_text(encoding="utf-8"), content)

    def test_load_non_dict_root_or_section_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [
                ("array.json", '["list"]'),
                ("bad_section.json", '{"server": "bad"}'),
            ]:
                with self.subTest(name=name):
                    cfg_path = Path(tmpdir) / name
                    cfg_path.write_text(content, encoding="utf-8")
                    store = ConfigStore(cfg_path)
                    with self.assertRaises(ConfigError) as ctx:
                        store.load()
                    self.assertEqual(ctx.exception.code, "CONFIG_INVALID_TYPE")

    def test_save_writes_atomically_and_leaves_no_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            payload = copy.deepcopy(DEFAULT_CONFIG)
            payload["server"]["port"] = 8123

            saved = store.save(payload)
            self.assertEqual(saved["server"]["port"], 8123)
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

            # Returned value must be a deep copy
            saved["server"]["port"] = 1111
            self.assertEqual(store.load()["server"]["port"], 8123)

    def test_save_incomplete_or_invalid_config_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()
            before = cfg_path.read_text(encoding="utf-8")

            with self.assertRaises(ConfigError) as ctx1:
                store.save({})
            self.assertEqual(ctx1.exception.code, "CONFIG_VALIDATION_ERROR")

            bad_payload = copy.deepcopy(DEFAULT_CONFIG)
            bad_payload["server"]["port"] = 70000
            with self.assertRaises(ConfigError) as ctx2:
                store.save(bad_payload)
            self.assertEqual(ctx2.exception.code, "CONFIG_VALIDATION_ERROR")

            self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)

    def test_save_temp_cleanup_on_os_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()
            before = cfg_path.read_text(encoding="utf-8")

            updated = copy.deepcopy(DEFAULT_CONFIG)
            updated["server"]["port"] = 8888

            with patch("backend.config_store.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(ConfigError) as ctx:
                    store.save(updated)
            self.assertEqual(ctx.exception.code, "CONFIG_IO_ERROR")
            self.assertEqual(ctx.exception.http_status, 500)
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

    def test_update_deep_merges_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()

            res1 = store.update(
                {
                    "llama": {"temperature": 0.65},
                    "t2i": {"enabled": True, "plugin_settings": {"stub_a": 1}},
                }
            )
            self.assertEqual(res1["llama"]["temperature"], 0.65)
            self.assertEqual(res1["llama"]["ctx_size"], DEFAULT_CONFIG["llama"]["ctx_size"])
            self.assertTrue(res1["t2i"]["enabled"])
            self.assertEqual(res1["t2i"]["plugin_settings"], {"stub_a": 1})

            res2 = store.update({"t2i": {"plugin_settings": {"stub_b": "ok"}}})
            self.assertEqual(res2["t2i"]["plugin_settings"], {"stub_a": 1, "stub_b": "ok"})
            self.assertTrue(res2["t2i"]["enabled"])

    def test_update_validation_failure_does_not_mutate_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Case 1: file does not exist yet -> invalid update must not create file
            missing_path = Path(tmpdir) / "missing.json"
            store_missing = ConfigStore(missing_path)
            with self.assertRaises(ConfigError):
                store_missing.update({"server": {"port": 70000}})
            self.assertFalse(missing_path.exists())

            # Case 2: file exists -> invalid update must preserve exact contents
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()
            before = cfg_path.read_text(encoding="utf-8")

            with self.assertRaises(ConfigError) as ctx:
                store.update(["not", "a", "dict"])  # type: ignore[arg-type]
            self.assertEqual(ctx.exception.code, "CONFIG_INVALID_TYPE")
            self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)

    def test_validation_matrix_types_and_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()
            before = cfg_path.read_text(encoding="utf-8")

            invalid_patches = [
                ("server.host empty", {"server": {"host": "   "}}),
                ("server.port 0", {"server": {"port": 0}}),
                ("server.port 65536", {"server": {"port": 65536}}),
                ("server.port str", {"server": {"port": "7860"}}),
                ("server.port bool", {"server": {"port": True}}),
                ("server.open_browser str", {"server": {"open_browser": "yes"}}),
                ("skill.max_prompt_chars -1", {"skill": {"max_prompt_chars": -1}}),
                ("skill.max_prompt_chars bool", {"skill": {"max_prompt_chars": True}}),
                ("llama.engine_mode unknown", {"llama": {"engine_mode": "unknown"}}),
                ("llama.server_url bad", {"llama": {"server_url": "not-a-url"}}),
                ("llama.server_url ftp", {"llama": {"server_url": "ftp://127.0.0.1:8080"}}),
                ("llama.server_url spaces", {"llama": {"server_url": "   "}}),
                ("llama.host empty", {"llama": {"host": ""}}),
                ("llama.port 0", {"llama": {"port": 0}}),
                ("llama.port bool", {"llama": {"port": False}}),
                ("llama.ctx_size 0", {"llama": {"ctx_size": 0}}),
                ("llama.ctx_size bool", {"llama": {"ctx_size": True}}),
                ("llama.n_gpu_layers -2", {"llama": {"n_gpu_layers": -2}}),
                ("llama.n_gpu_layers bool", {"llama": {"n_gpu_layers": True}}),
                ("llama.startup_timeout_sec 0", {"llama": {"startup_timeout_sec": 0}}),
                ("llama.request_timeout_sec -1", {"llama": {"request_timeout_sec": -1}}),
                ("llama.temperature -0.1", {"llama": {"temperature": -0.1}}),
                ("llama.temperature 2.1", {"llama": {"temperature": 2.1}}),
                ("llama.temperature bool", {"llama": {"temperature": False}}),
                ("llama.temperature nan", {"llama": {"temperature": float("nan")}}),
                ("llama.temperature inf", {"llama": {"temperature": float("inf")}}),
                ("llama.top_p -0.01", {"llama": {"top_p": -0.01}}),
                ("llama.top_p 1.2", {"llama": {"top_p": 1.2}}),
                ("llama.top_p bool", {"llama": {"top_p": True}}),
                ("llama.max_tokens 0", {"llama": {"max_tokens": 0}}),
                ("llama.max_tokens bool", {"llama": {"max_tokens": False}}),
                ("llama.max_history_turns 0", {"llama": {"max_history_turns": 0}}),
                ("t2i.enabled str", {"t2i": {"enabled": "false"}}),
                ("t2i.active_plugin empty", {"t2i": {"active_plugin": "  "}}),
                ("t2i.context_window_turns 0", {"t2i": {"context_window_turns": 0}}),
                ("t2i.plugins_dir empty", {"t2i": {"plugins_dir": ""}}),
                ("t2i.plugin_settings list", {"t2i": {"plugin_settings": []}}),
            ]
            for label, patch_data in invalid_patches:
                with self.subTest(label=label):
                    with self.assertRaises(ConfigError) as ctx:
                        store.update(patch_data)
                    err_dict = ctx.exception.to_dict()
                    self.assertIn("code", err_dict)
                    self.assertIn("message", err_dict)
                    self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)

            # Valid boundary values must pass
            valid_patch = {
                "skill": {"max_prompt_chars": 0},
                "llama": {
                    "engine_mode": "cli",
                    "server_url": "",
                    "n_gpu_layers": -1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "ctx_size": 1,
                    "max_tokens": 1,
                    "max_history_turns": 1,
                },
                "t2i": {"context_window_turns": 1},
            }
            updated = store.update(valid_patch)
            self.assertEqual(updated["llama"]["engine_mode"], "cli")
            self.assertEqual(updated["llama"]["n_gpu_layers"], -1)

    def test_concurrent_updates_are_thread_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()

            def worker(idx: int) -> None:
                store.update(
                    {
                        "t2i": {"plugin_settings": {f"k_{idx}": idx}},
                    }
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(worker, i) for i in range(24)]
                for fut in futures:
                    fut.result()

            final = store.load()
            for i in range(24):
                self.assertEqual(final["t2i"]["plugin_settings"][f"k_{i}"], i)
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

    def test_server_url_validation_rejects_malformed_and_accepts_valid_urls(self) -> None:
        malformed_urls = [
            "http://:8080",
            "http://bad host:8080",
            "http://127.0.0.1:abc",
            "http://127.0.0.1:99999",
            "http://127.0.0.1:8080/a\nb",
            "http://127.0.0.1:8080/a b",
            "http://[::1:8080",
            "http://::1]:8080",
            "http://[invalid",
            "http://127.0.0.1:8080]",
            "http://[not-ipv6]:8080",
            "http://127.0.0.1:",
            "http://[::1]:",
            "http://user@127.0.0.1:",
        ]
        valid_urls = [
            "",
            "http://127.0.0.1:8080",
            "https://localhost:8443/v1?a=1#frag",
            "http://[::1]:8080",
            "http://[::1]",
            "http://127.0.0.1",
            "http://user:pass@127.0.0.1:8080",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            store = ConfigStore(cfg_path)
            store.load()
            before = cfg_path.read_text(encoding="utf-8")

            for bad_url in malformed_urls:
                with self.subTest(bad_url=bad_url):
                    with self.assertRaises(ConfigError) as ctx:
                        store.update({"llama": {"server_url": bad_url}})
                    self.assertEqual(ctx.exception.code, "CONFIG_VALIDATION_ERROR")
                    self.assertEqual(ctx.exception.field, "llama.server_url")
                    self.assertEqual(ctx.exception.http_status, 400)
                    self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)

            for valid_url in valid_urls:
                with self.subTest(valid_url=valid_url):
                    updated = store.update({"llama": {"server_url": valid_url}})
                    self.assertEqual(updated["llama"]["server_url"], valid_url)

    def test_blank_example_path_raises_config_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            for blank in ("", "   "):
                with self.subTest(blank=blank):
                    with self.assertRaises(ConfigError) as ctx:
                        ConfigStore(cfg_path, example_path=blank)
                    self.assertEqual(ctx.exception.code, "CONFIG_VALIDATION_ERROR")
                    self.assertEqual(ctx.exception.field, "example_path")
                    self.assertFalse(cfg_path.exists())


if __name__ == "__main__":
    unittest.main()
