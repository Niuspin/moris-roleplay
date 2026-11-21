"""Unit and real-fixture tests for backend.skill_loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.skill_loader import SkillLoader, SkillLoadError, SkillLoadResult

FIREFLY_DIR = Path(r"C:\Users\nzp18\Desktop\角色skill\流萤\firefly-skill")
YAE_MIKO_FILE = Path(r"C:\Users\nzp18\Desktop\角色skill\八重神子\SKILL.md")


class SkillLoaderTest(unittest.TestCase):
    @unittest.skipUnless(FIREFLY_DIR.exists(), "Real firefly-skill fixture not present")
    def test_real_firefly_skill_all_files(self) -> None:
        res = SkillLoader.load(FIREFLY_DIR, exclude_meta_docs=False)
        self.assertIsInstance(res, SkillLoadResult)
        self.assertEqual(res.source_type, "directory")
        self.assertEqual(res.character_name, "firefly-hsr")
        self.assertTrue(res.description)
        self.assertEqual(
            res.loaded_files,
            [
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
            ],
        )
        self.assertEqual(res.ignored_files, ["LICENSE", "manifest.json"])
        self.assertEqual(res.excluded_files, [])
        self.assertFalse(any(".git" in item for item in res.loaded_files + res.ignored_files))
        self.assertNotIn("license: MIT", res.system_prompt)
        # Internal --- sections in conflicts.md must be preserved
        self.assertIn("## [开拓者关系]", res.system_prompt)
        self.assertIn("## [完全燃烧]", res.system_prompt)
        self.assertIn("\n---\n", res.system_prompt)
        self.assertEqual(res.total_chars, len(res.system_prompt))
        self.assertGreater(res.total_chars, 20000)
        self.assertGreater(res.estimated_tokens, 0)
        self.assertTrue(res.warnings)

    @unittest.skipUnless(FIREFLY_DIR.exists(), "Real firefly-skill fixture not present")
    def test_real_firefly_skill_exclude_meta_docs(self) -> None:
        res = SkillLoader.load(FIREFLY_DIR, exclude_meta_docs=True)
        self.assertEqual(
            res.loaded_files,
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
            res.excluded_files,
            ["README.md", "quality-report.md", "test-report.md"],
        )
        self.assertEqual(res.ignored_files, ["LICENSE", "manifest.json"])
        self.assertNotIn("<!-- module: README.md -->", res.system_prompt)

    @unittest.skipUnless(YAE_MIKO_FILE.exists(), "Real 八重神子/SKILL.md fixture not present")
    def test_real_yae_miko_single_file(self) -> None:
        res = SkillLoader.load(YAE_MIKO_FILE)
        self.assertEqual(res.source_type, "file")
        self.assertEqual(res.character_name, "yae-miko-perspective")
        self.assertIn("鸣神大社宫司", res.description)
        self.assertIn("八重宫司模式", res.description)
        self.assertEqual(res.loaded_files, ["SKILL.md"])
        self.assertEqual(res.ignored_files, [])
        self.assertEqual(res.excluded_files, [])
        self.assertNotIn("name: yae-miko-perspective", res.system_prompt)
        self.assertIn("\n---\n", res.system_prompt)
        self.assertIn("## 回答工作流", res.system_prompt)
        self.assertEqual(res.total_chars, len(res.system_prompt))

    def test_uppercase_md_and_other_md_alphabetical(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "SKILL.MD").write_text("---\nname: Upper\n---\nCore", encoding="utf-8")
            (root / "zebra.MD").write_text("Zebra content", encoding="utf-8")
            (root / "alpha.md").write_text("Alpha content", encoding="utf-8")
            (root / "CHANGELOG.md").write_text("Changes", encoding="utf-8")
            (root / "readme.txt").write_text("Ignore me", encoding="utf-8")

            res = SkillLoader.load(root)
            self.assertEqual(res.character_name, "Upper")
            self.assertEqual(res.loaded_files, ["SKILL.MD", "alpha.md", "zebra.MD", "CHANGELOG.md"])
            self.assertEqual(res.ignored_files, ["readme.txt"])

    def test_utf8_bom_and_crlf_frontmatter_and_internal_hr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "hero.md"
            raw = (
                "\ufeff---\r\n"
                "name: bom-hero\r\n"
                'description: "Line \\"quoted\\" text"\r\n'
                "---\r\n"
                "Body\r\n"
                "---\r\n"
                "Keep\r\n"
            )
            md_path.write_bytes(raw.encode("utf-8"))

            res = SkillLoader.load(md_path)
            self.assertEqual(res.character_name, "bom-hero")
            self.assertEqual(res.description, 'Line "quoted" text')
            self.assertEqual(res.system_prompt, "Body\n---\nKeep")
            self.assertNotIn("\r", res.system_prompt)

    def test_yaml_block_scalars_folded_and_literal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "folded.md"
            md_path.write_text(
                "---\n"
                "name: folded-hero\n"
                "description: >\n"
                "  First line\n"
                "  second line\n"
                "\n"
                "  New paragraph\n"
                "---\n"
                "Prompt body\n",
                encoding="utf-8",
            )
            res = SkillLoader.load(md_path)
            self.assertEqual(res.character_name, "folded-hero")
            self.assertEqual(res.description, "First line second line\n\nNew paragraph")
            self.assertEqual(res.system_prompt, "Prompt body")

    def test_leading_hr_without_yaml_keys_not_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "plain.md"
            md_path.write_text("---\n# Heading\n---\nBody", encoding="utf-8")

            res = SkillLoader.load(md_path)
            self.assertEqual(res.character_name, "plain")
            self.assertEqual(res.system_prompt, "---\n# Heading\n---\nBody")

    def test_leading_hr_with_bare_url_or_timestamp_not_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [
                ("url_hr.md", "---\nhttp://example.com/path\n---\nBody"),
                ("time_hr.md", "---\n12:30\n---\nBody"),
            ]:
                with self.subTest(name=name):
                    md_path = Path(tmpdir) / name
                    md_path.write_text(content, encoding="utf-8")
                    res = SkillLoader.load(md_path)
                    self.assertEqual(res.character_name, md_path.stem)
                    self.assertEqual(res.system_prompt, content)

    def test_frontmatter_empty_value_and_json_metadata_with_block_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "meta_empty.md"
            md_path.write_text(
                "---\n"
                "name: meta-hero\n"
                "description:\n"
                'metadata: {"kit":"v1"}\n'
                "notes: |\n"
                "  Line 1\n"
                "  Line 2\n"
                "---\n"
                "Actual prompt\n",
                encoding="utf-8",
            )
            res = SkillLoader.load(md_path)
            self.assertEqual(res.character_name, "meta-hero")
            self.assertEqual(res.description, "")
            self.assertEqual(res.system_prompt, "Actual prompt")

    def test_empty_and_whitespace_md_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "SKILL.md").write_text("Main prompt", encoding="utf-8")
            (root / "empty.md").write_text("   \n\t\n", encoding="utf-8")

            res = SkillLoader.load(root)
            self.assertEqual(res.loaded_files, ["SKILL.md", "empty.md"])
            self.assertIn("Main prompt", res.system_prompt)
            self.assertTrue(any("empty.md" in w for w in res.warnings))

            only_empty = root / "only_empty.md"
            only_empty.write_text("---\nname: blank\n---\n   \n", encoding="utf-8")
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(only_empty)
            self.assertEqual(ctx.exception.code, "SKILL_EMPTY_CONTENT")

    def test_folder_only_non_md_raises_without_reading_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "LICENSE").write_text("MIT", encoding="utf-8")
            # Invalid UTF-8 in a non-.md file must never be decoded or cause SKILL_ENCODING_ERROR
            (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(root)
            self.assertEqual(ctx.exception.code, "SKILL_NO_MARKDOWN_FILES")

    def test_empty_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(tmpdir)
            self.assertEqual(ctx.exception.code, "SKILL_EMPTY_DIRECTORY")

    def test_single_non_md_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "notes.txt"
            txt_path.write_text("hello", encoding="utf-8")
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(txt_path)
            self.assertEqual(ctx.exception.code, "SKILL_UNSUPPORTED_FILE")

    def test_nonexistent_and_blank_path_raises(self) -> None:
        for bad_path, expected_code in [
            ("", "SKILL_PATH_EMPTY"),
            ("   ", "SKILL_PATH_EMPTY"),
            (r"C:\nonexistent\path\that\does\not\exist\skill.md", "SKILL_PATH_NOT_FOUND"),
        ]:
            with self.subTest(bad_path=bad_path):
                with self.assertRaises(SkillLoadError) as ctx:
                    SkillLoader.load(bad_path)
                self.assertEqual(ctx.exception.code, expected_code)

    def test_git_dir_trap_md_never_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            git_dir = root / ".git"
            git_dir.mkdir()
            (git_dir / "trap.md").write_bytes(b"\xff\xfeTRAP_SECRET")
            (root / ".hidden.md").write_bytes(b"\xff\xfeHIDDEN_SECRET")

            # Folder with only .git/trap.md and .hidden.md raises SKILL_NO_MARKDOWN_FILES
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(root)
            self.assertEqual(ctx.exception.code, "SKILL_NO_MARKDOWN_FILES")

            # With valid SKILL.md, .git/trap.md and .hidden.md are never read or listed
            (root / "SKILL.md").write_text("Genuine skill", encoding="utf-8")
            res = SkillLoader.load(root)
            self.assertEqual(res.loaded_files, ["SKILL.md"])
            self.assertEqual(res.ignored_files, [])
            self.assertNotIn("TRAP_SECRET", res.system_prompt)
            self.assertNotIn("HIDDEN_SECRET", res.system_prompt)

    def test_top_level_precedence_and_nested_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "bundle"
            nested.mkdir()
            (nested / "SKILL.md").write_text("---\nname: NestedHero\n---\nNested body", encoding="utf-8")
            (nested / "notes.txt").write_text("skip", encoding="utf-8")

            # 0 top-level .md -> falls back to non-dot subdirectory
            res_nested = SkillLoader.load(root)
            self.assertEqual(res_nested.character_name, "NestedHero")
            self.assertEqual(res_nested.loaded_files, ["bundle/SKILL.md"])
            self.assertEqual(res_nested.ignored_files, ["bundle/notes.txt"])

            # Once top-level .md exists, top-level scan takes precedence
            (root / "SKILL.md").write_text("---\nname: TopHero\n---\nTop body", encoding="utf-8")
            res_top = SkillLoader.load(root)
            self.assertEqual(res_top.character_name, "TopHero")
            self.assertEqual(res_top.loaded_files, ["SKILL.md"])
            self.assertNotIn("Nested body", res_top.system_prompt)

    def test_exclude_meta_docs_all_meta_raises_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("Docs only", encoding="utf-8")
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(root, exclude_meta_docs=True)
            self.assertEqual(ctx.exception.code, "SKILL_EMPTY_CONTENT")

    def test_max_prompt_chars_truncation_and_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "long.md"
            md_path.write_text("A" * 1000, encoding="utf-8")

            res = SkillLoader.load(md_path, max_prompt_chars=150)
            self.assertLessEqual(len(res.system_prompt), 150)
            self.assertEqual(res.total_chars, len(res.system_prompt))
            self.assertTrue(any("truncated" in w.lower() for w in res.warnings))

            # Tiny limit smaller than suffix length still obeys max_prompt_chars
            res_tiny = SkillLoader.load(md_path, max_prompt_chars=10)
            self.assertEqual(len(res_tiny.system_prompt), 10)
            self.assertEqual(res_tiny.total_chars, 10)

            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(md_path, max_prompt_chars=-1)
            self.assertEqual(ctx.exception.code, "SKILL_INVALID_ARGUMENT")

    def test_invalid_utf8_bytes_raises_skill_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_md = Path(tmpdir) / "bad.md"
            bad_md.write_bytes(b"\xff\xfe\x80\x81")
            with self.assertRaises(SkillLoadError) as ctx:
                SkillLoader.load(bad_md)
            self.assertEqual(ctx.exception.code, "SKILL_ENCODING_ERROR")
            self.assertIn("code", ctx.exception.to_dict())

    def test_class_and_instance_invocation_plus_to_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "character.md"
            md_path.write_text("Speak politely.", encoding="utf-8")

            r_cls = SkillLoader.load(md_path)
            r_inst = SkillLoader().load(md_path)
            self.assertEqual(r_cls.to_dict(), r_inst.to_dict())
            serialized = json.dumps(r_cls.to_dict(), ensure_ascii=False)
            self.assertIn("character", serialized)


if __name__ == "__main__":
    unittest.main()
