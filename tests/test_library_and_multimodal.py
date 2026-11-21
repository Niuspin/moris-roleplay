"""Unit and integration tests for LibraryManager (engines, models, skills) and Multimodal Vision support."""

from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from backend.library_manager import LibraryManager
from backend.llama_engine import LlamaServerEngine
from backend.server import build_application_server
from e2e_tests.harness import (
    FakeLlamaServer,
    build_mock_llama_exe,
    find_free_port,
    http_request,
    read_mock_llama_args_log,
    write_dummy_gguf,
)


class LibraryAndMultimodalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="moris_lib_mm_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_dir, ignore_errors=True))

    def test_library_manager_scans_engines_multimodal_models_and_skills(self) -> None:
        fake_repo = self.tmp_dir / "repo"
        mgr = LibraryManager(repo_root=fake_repo, include_desktop_skills=True)
        mgr.ensure_default_directories()

        # 1. Populate engines/llama.cpp/
        eng_bin = fake_repo / "engines" / "llama.cpp" / "llama-server.exe"
        eng_bin.parent.mkdir(parents=True, exist_ok=True)
        eng_bin.write_bytes(b"MZ_FAKE_EXE")

        # 2. Populate models/llm/ and models/multimodal/ with main GGUF + mmproj GGUF
        write_dummy_gguf(fake_repo / "models" / "llm" / "qwen2.5-7b-instruct.gguf")
        mm_main = write_dummy_gguf(
            fake_repo / "models" / "multimodal" / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        )
        mm_proj = write_dummy_gguf(
            fake_repo / "models" / "multimodal" / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
        )

        # 3. Populate skills/
        custom_skill_dir = fake_repo / "skills" / "custom-char"
        custom_skill_dir.mkdir(parents=True, exist_ok=True)
        (custom_skill_dir / "SKILL.md").write_text(
            "---\nname: CustomLocalChar\ndescription: Local library character\n---\n# Custom Rule",
            encoding="utf-8",
        )
        (custom_skill_dir / "ignore.json").write_text("{}", encoding="utf-8")

        overview = mgr.scan_all({})

        # Verify engines scan
        eng_items = overview["engines"]["items"]
        self.assertEqual(len(eng_items), 1)
        self.assertEqual(eng_items[0]["name"], "llama-server.exe")
        self.assertEqual(eng_items[0]["kind"], "server")
        self.assertTrue(eng_items[0]["supports_multimodal"])

        # Verify models scan & automatic multimodal pairing
        models_list = overview["models"]["models"]
        mmprojs_list = overview["models"]["mmprojs"]
        self.assertEqual(len(models_list), 2)
        self.assertEqual(len(mmprojs_list), 1)
        self.assertEqual(mmprojs_list[0]["path"], str(mm_proj.resolve()))

        vl_entry = next(m for m in models_list if "VL" in m["name"])
        self.assertEqual(vl_entry["modality"], "multimodal")
        self.assertEqual(vl_entry["suggested_mmproj"], str(mm_proj.resolve()))
        self.assertEqual(vl_entry["path"], str(mm_main.resolve()))

        # Verify skills scan (in-project custom-char + desktop skills)
        skill_items = overview["skills"]["items"]
        char_names = {s["character_name"] for s in skill_items}
        self.assertIn("CustomLocalChar", char_names)
        self.assertIn("firefly-hsr", char_names)
        self.assertIn("yae-miko-perspective", char_names)

    def test_managed_server_forwards_mmproj_cli_argument(self) -> None:
        exe = build_mock_llama_exe(self.tmp_dir / "bin", "llama-server.exe")
        model = write_dummy_gguf(self.tmp_dir / "models" / "vlm-main.gguf")
        mmproj = write_dummy_gguf(self.tmp_dir / "models" / "mmproj-vlm-f16.gguf")
        args_log = self.tmp_dir / "mm_args.jsonl"
        port = find_free_port()

        import os
        from unittest.mock import patch

        eng = LlamaServerEngine(
            {
                "engine_mode": "managed_server",
                "llama_cpp_path": str(exe),
                "model_path": str(model),
                "mmproj_path": str(mmproj),
                "port": port,
                "startup_timeout_sec": 10,
            }
        )
        self.addCleanup(eng.shutdown)
        with patch.dict(os.environ, {"MOCK_LLAMA_ARGS_LOG": str(args_log)}):
            eng.ensure_ready()

        logged = read_mock_llama_args_log(args_log)
        self.assertEqual(len(logged), 1)
        argv = logged[0]
        self.assertIn("--mmproj", argv)
        mm_idx = argv.index("--mmproj")
        self.assertEqual(argv[mm_idx + 1], str(mmproj.resolve()))

    def test_server_library_endpoints_and_multimodal_image_chat(self) -> None:
        fake_llama = FakeLlamaServer(default_reply="我看到了你发来的图片！").start()
        self.addCleanup(fake_llama.stop)

        cfg_path = self.tmp_dir / "config.json"
        srv = build_application_server(
            config_path=cfg_path,
            host="127.0.0.1",
            port=find_free_port(),
        )
        port = int(srv.server_address[1])
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()

        def _cleanup() -> None:
            try:
                srv.shutdown()
            except Exception:
                pass
            try:
                srv.chat_service.shutdown()
            except Exception:
                pass
            try:
                srv.server_close()
            except Exception:
                pass
            t.join(timeout=2.0)

        self.addCleanup(_cleanup)
        base_url = f"http://127.0.0.1:{port}"

        # 1. GET /api/library
        lib_resp = http_request(f"{base_url}/api/library")
        self.assertEqual(lib_resp.status, 200)
        self.assertIn("engines", lib_resp.data)
        self.assertIn("models", lib_resp.data)
        self.assertIn("skills", lib_resp.data)

        # 2. Configure external FakeLlamaServer and send multimodal chat with image
        http_request(
            f"{base_url}/api/config",
            method="PUT",
            json_body={
                "llama": {
                    "engine_mode": "external_server",
                    "server_url": fake_llama.base_url,
                }
            },
        )
        tiny_png_data_url = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        chat_resp = http_request(
            f"{base_url}/api/chat",
            method="POST",
            json_body={
                "session_id": "mm-sess",
                "message": "请描述这张图片",
                "images": [tiny_png_data_url],
            },
        )
        self.assertEqual(chat_resp.status, 200)
        self.assertEqual(chat_resp.data["reply"], "我看到了你发来的图片！")

        sent_messages = fake_llama.last_request["body"]["messages"]
        user_msg = sent_messages[-1]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0], {"type": "text", "text": "请描述这张图片"})
        self.assertEqual(
            user_msg["content"][1],
            {"type": "image_url", "image_url": {"url": tiny_png_data_url}},
        )


if __name__ == "__main__":
    unittest.main()
