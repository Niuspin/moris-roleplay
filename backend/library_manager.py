"""Dedicated local library manager for llama.cpp engines, GGUF/multimodal models, and character skills."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.skill_loader import META_INDEX, _parse_simple_yaml, _split_frontmatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINES_DIR = REPO_ROOT / "engines"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
DESKTOP_SKILLS_DIR = Path(r"C:\Users\nzp18\Desktop\角色skill")

_SERVER_EXE_NAMES = frozenset(
    {
        "llama-server.exe",
        "server.exe",
        "llama-server.cmd",
        "llama-server.bat",
        "llama-server",
        "server",
    }
)

_CLI_EXE_NAMES = frozenset(
    {
        "llama-cli.exe",
        "main.exe",
        "llama-cli.cmd",
        "llama-cli.bat",
        "llama-cli",
        "main",
    }
)

_MTMD_CLI_EXE_NAMES = frozenset(
    {
        "llama-mtmd-cli.exe",
        "llama-llava-cli.exe",
        "llama-qwen2vl-cli.exe",
        "llama-minicpmv-cli.exe",
        "llama-gemma3-cli.exe",
        "llama-mtmd-cli",
        "llama-llava-cli",
    }
)

_MMPROJ_KEYWORDS = ("mmproj", "clip", "vision-proj", "vision_proj", "projector")
_VISION_MODEL_KEYWORDS = (
    "-vl",
    "_vl",
    "vision",
    "llava",
    "minicpm-v",
    "minicpmv",
    "pixtral",
    "internvl",
    "gemma-3",
    "gemma3",
    "moondream",
    "qwen2-vl",
    "qwen2.5-vl",
)


def _is_mmproj_filename(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in _MMPROJ_KEYWORDS)


def _is_vision_model_name(name: str, parent_parts: tuple[str, ...]) -> bool:
    lower = name.lower()
    if any(kw in lower for kw in _VISION_MODEL_KEYWORDS):
        return True
    return any("multimodal" in part.lower() or "vision" in part.lower() for part in parent_parts)


def _format_size_mb(size_bytes: int) -> float:
    return round(size_bytes / (1024.0 * 1024.0), 2)


def _resolve_dir_candidates(raw_dirs: list[str | Path], base_root: Path = REPO_ROOT) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for item in raw_dirs:
        if not item:
            continue
        p = Path(item).expanduser()
        if not p.is_absolute():
            p = base_root / p
        try:
            norm = str(p.resolve())
        except OSError:
            continue
        key = norm.lower() if os.name == "nt" else norm
        if key not in seen and p.exists() and p.is_dir():
            seen.add(key)
            resolved.append(p.resolve())
    return resolved


def _iter_files_up_to_depth(root: Path, max_depth: int = 3) -> list[Path]:
    results: list[Path] = []

    def _walk(curr: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(curr.iterdir(), key=lambda e: e.name.lower())
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                results.append(entry)
            elif entry.is_dir() and depth < max_depth:
                _walk(entry, depth + 1)

    _walk(root, 0)
    return results


def _peek_md_metadata(md_path: Path, fallback_name: str) -> tuple[str, str]:
    """Quickly read the top of an .md file to extract YAML frontmatter name & description."""
    try:
        raw = md_path.read_bytes()[:16384]
        text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        fm, body = _split_frontmatter(text)
        name = (fm.get("name") or "").strip() or fallback_name
        desc = (fm.get("description") or "").strip()
        if not desc:
            for line in body.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    desc = stripped[:140]
                    break
        return name, desc
    except OSError:
        return fallback_name, ""


class LibraryManager:
    """Scans and manages dedicated local directories for llama.cpp engines, models, and skills."""

    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        *,
        include_desktop_skills: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.engines_dir = self.repo_root / "engines"
        self.models_dir = self.repo_root / "models"
        self.skills_dir = self.repo_root / "skills"
        self.include_desktop_skills = include_desktop_skills

    def ensure_default_directories(self) -> None:
        """Ensure in-project engines/, models/, and skills/ directories exist."""
        for d in (
            self.engines_dir,
            self.engines_dir / "llama.cpp",
            self.models_dir,
            self.models_dir / "llm",
            self.models_dir / "multimodal",
            self.skills_dir,
        ):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def scan_engines(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Scan local engines/ directory and configured llama_cpp_path for binaries."""
        llama_cfg = (config or {}).get("llama", {}) if isinstance(config, dict) else {}
        cfg_engine_dir = str(llama_cfg.get("engine_dir", "engines") or "engines").strip()
        active_path = str(llama_cfg.get("llama_cpp_path", "") or "").strip()

        raw_roots: list[str | Path] = [self.engines_dir, cfg_engine_dir]
        if active_path:
            ap = Path(active_path).expanduser()
            if ap.is_dir():
                raw_roots.append(ap)
            elif ap.is_file():
                raw_roots.append(ap.parent)

        roots = _resolve_dir_candidates(raw_roots, base_root=self.repo_root)
        items: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for root in roots:
            for file_path in _iter_files_up_to_depth(root, max_depth=3):
                name_lower = file_path.name.lower()
                kind: str | None = None
                supports_mm = False
                if name_lower in _SERVER_EXE_NAMES:
                    kind = "server"
                    supports_mm = True
                elif name_lower in _MTMD_CLI_EXE_NAMES:
                    kind = "multimodal_cli"
                    supports_mm = True
                elif name_lower in _CLI_EXE_NAMES:
                    kind = "cli"
                    supports_mm = False

                if kind is None:
                    continue

                resolved_str = str(file_path.resolve())
                norm_key = resolved_str.lower() if os.name == "nt" else resolved_str
                if norm_key in seen_paths:
                    continue
                seen_paths.add(norm_key)

                try:
                    size_mb = _format_size_mb(file_path.stat().st_size)
                except OSError:
                    size_mb = 0.0

                items.append(
                    {
                        "name": file_path.name,
                        "path": resolved_str,
                        "directory": str(file_path.parent.resolve()),
                        "kind": kind,
                        "supports_multimodal": supports_mm,
                        "size_mb": size_mb,
                    }
                )

        # Sort server binaries first, then multimodal_cli, then cli
        kind_priority = {"server": 0, "multimodal_cli": 1, "cli": 2}
        items.sort(key=lambda x: (kind_priority.get(x["kind"], 9), x["name"].lower()))

        return {
            "default_dir": str(self.engines_dir),
            "scanned_dirs": [str(r) for r in roots],
            "active_path": active_path,
            "items": items,
        }

    def scan_models(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Scan local models/ directory for .gguf LLM/VLM models and mmproj vision projectors."""
        llama_cfg = (config or {}).get("llama", {}) if isinstance(config, dict) else {}
        cfg_models_dir = str(llama_cfg.get("models_dir", "models") or "models").strip()
        active_model = str(llama_cfg.get("model_path", "") or "").strip()
        active_mmproj = str(llama_cfg.get("mmproj_path", "") or "").strip()

        raw_roots: list[str | Path] = [self.models_dir, cfg_models_dir]
        for active_file in (active_model, active_mmproj):
            if active_file:
                af = Path(active_file).expanduser()
                if af.is_file():
                    raw_roots.append(af.parent)
                elif af.is_dir():
                    raw_roots.append(af)

        roots = _resolve_dir_candidates(raw_roots, base_root=self.repo_root)
        main_models: list[dict[str, Any]] = []
        mmproj_models: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for root in roots:
            for file_path in _iter_files_up_to_depth(root, max_depth=3):
                if file_path.suffix.lower() != ".gguf":
                    continue
                try:
                    st = file_path.stat()
                except OSError:
                    continue
                if st.st_size <= 0:
                    continue

                resolved_str = str(file_path.resolve())
                norm_key = resolved_str.lower() if os.name == "nt" else resolved_str
                if norm_key in seen_paths:
                    continue
                seen_paths.add(norm_key)

                try:
                    rel_path = str(file_path.resolve().relative_to(root))
                except ValueError:
                    rel_path = file_path.name

                parent_parts = file_path.parts[-3:-1]
                size_mb = _format_size_mb(st.st_size)

                if _is_mmproj_filename(file_path.name):
                    mmproj_models.append(
                        {
                            "name": file_path.name,
                            "path": resolved_str,
                            "directory": str(file_path.parent.resolve()),
                            "rel_path": rel_path,
                            "role": "mmproj",
                            "size_mb": size_mb,
                        }
                    )
                else:
                    is_vision = _is_vision_model_name(file_path.name, parent_parts)
                    main_models.append(
                        {
                            "name": file_path.name,
                            "path": resolved_str,
                            "directory": str(file_path.parent.resolve()),
                            "rel_path": rel_path,
                            "role": "model",
                            "modality": "multimodal" if is_vision else "text",
                            "suggested_mmproj": "",
                            "size_mb": size_mb,
                        }
                    )

        # Pair main models with matching mmproj files in the same directory or by stem token match
        for model in main_models:
            same_dir_mmprojs = [
                m for m in mmproj_models if m["directory"] == model["directory"]
            ]
            if len(same_dir_mmprojs) == 1:
                model["suggested_mmproj"] = same_dir_mmprojs[0]["path"]
                model["modality"] = "multimodal"
            elif len(same_dir_mmprojs) > 1:
                model_stem_tokens = {
                    tok
                    for tok in model["name"].lower().replace("_", "-").split("-")
                    if len(tok) >= 3 and tok not in ("gguf", "instruct", "chat")
                }
                best_match = same_dir_mmprojs[0]["path"]
                best_score = -1
                for cand in same_dir_mmprojs:
                    cand_lower = cand["name"].lower()
                    score = sum(1 for tok in model_stem_tokens if tok in cand_lower)
                    if score > best_score:
                        best_score = score
                        best_match = cand["path"]
                model["suggested_mmproj"] = best_match
                model["modality"] = "multimodal"
            elif model["modality"] == "multimodal" and len(mmproj_models) == 1:
                model["suggested_mmproj"] = mmproj_models[0]["path"]

        main_models.sort(key=lambda m: (0 if m["modality"] == "multimodal" else 1, m["name"].lower()))
        mmproj_models.sort(key=lambda m: m["name"].lower())

        return {
            "default_dir": str(self.models_dir),
            "scanned_dirs": [str(r) for r in roots],
            "active_model_path": active_model,
            "active_mmproj_path": active_mmproj,
            "models": main_models,
            "mmprojs": mmproj_models,
        }

    def _inspect_skill_dir(
        self,
        skill_dir: Path,
        library_label: str,
        parent_display_hint: str = "",
    ) -> dict[str, Any] | None:
        try:
            entries = sorted(skill_dir.iterdir(), key=lambda e: e.name.lower())
        except OSError:
            return None

        md_files: list[Path] = []
        ignored_count = 0
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                if entry.suffix.lower() == ".md":
                    md_files.append(entry)
                else:
                    ignored_count += 1

        if not md_files:
            return None

        # Ignore a folder that only has a placeholder README.md (like skills/README.md)
        if len(md_files) == 1 and md_files[0].name.lower() == "readme.md":
            return None

        primary_md = next(
            (f for f in md_files if f.name.lower() == "skill.md"),
            md_files[0],
        )
        fallback_name = parent_display_hint or skill_dir.name
        char_name, desc = _peek_md_metadata(primary_md, fallback_name=fallback_name)

        folder_label = parent_display_hint or skill_dir.name
        if folder_label and folder_label.lower() != char_name.lower():
            display_name = f"{folder_label} ({char_name})"
        else:
            display_name = char_name

        # If folder has only 1 .md file and 0 other files, we can still load the folder or file
        target_path = str(skill_dir.resolve())
        return {
            "display_name": display_name,
            "character_name": char_name,
            "description": desc,
            "path": target_path,
            "source_type": "directory",
            "library_label": library_label,
            "md_file_count": len(md_files),
            "ignored_file_count": ignored_count,
        }

    def scan_skills(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Scan in-project skills/ directory and external configured skill library roots."""
        skill_cfg = (config or {}).get("skill", {}) if isinstance(config, dict) else {}
        active_skill_path = str(skill_cfg.get("last_path", "") or "").strip()

        raw_dirs: list[str | Path] = [self.skills_dir]
        cfg_dirs = skill_cfg.get("library_dirs")
        if isinstance(cfg_dirs, list):
            for d in cfg_dirs:
                if isinstance(d, str) and d.strip():
                    raw_dirs.append(d.strip())
        if self.include_desktop_skills and DESKTOP_SKILLS_DIR.is_dir():
            raw_dirs.append(DESKTOP_SKILLS_DIR)

        roots = _resolve_dir_candidates(raw_dirs, base_root=self.repo_root)
        items: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for root in roots:
            is_builtin = root == self.skills_dir.resolve()
            lib_label = "项目内置库 (skills/)" if is_builtin else f"外部库 ({root.name})"

            try:
                top_entries = sorted(root.iterdir(), key=lambda e: e.name.lower())
            except OSError:
                continue

            for entry in top_entries:
                if entry.name.startswith("."):
                    continue

                if entry.is_file() and entry.suffix.lower() == ".md":
                    if entry.name.lower() in META_INDEX:
                        continue
                    resolved_str = str(entry.resolve())
                    norm_key = resolved_str.lower() if os.name == "nt" else resolved_str
                    if norm_key in seen_paths:
                        continue
                    seen_paths.add(norm_key)
                    char_name, desc = _peek_md_metadata(entry, fallback_name=entry.stem)
                    items.append(
                        {
                            "display_name": f"{entry.stem} ({char_name})"
                            if entry.stem.lower() != char_name.lower()
                            else char_name,
                            "character_name": char_name,
                            "description": desc,
                            "path": resolved_str,
                            "source_type": "file",
                            "library_label": lib_label,
                            "md_file_count": 1,
                            "ignored_file_count": 0,
                        }
                    )
                    continue

                if entry.is_dir():
                    # Check if entry itself is a skill folder containing .md files
                    direct_info = self._inspect_skill_dir(entry, lib_label, parent_display_hint=entry.name)
                    if direct_info is not None:
                        norm_key = (
                            direct_info["path"].lower()
                            if os.name == "nt"
                            else direct_info["path"]
                        )
                        if norm_key not in seen_paths:
                            seen_paths.add(norm_key)
                            items.append(direct_info)
                        continue

                    # Depth-2 check (e.g., 流萤/firefly-skill or 爱弥斯/Aemeath-skill)
                    try:
                        sub_entries = sorted(entry.iterdir(), key=lambda e: e.name.lower())
                    except OSError:
                        continue
                    for sub in sub_entries:
                        if sub.name.startswith(".") or not sub.is_dir():
                            continue
                        sub_info = self._inspect_skill_dir(
                            sub,
                            lib_label,
                            parent_display_hint=entry.name,
                        )
                        if sub_info is not None:
                            norm_key = (
                                sub_info["path"].lower()
                                if os.name == "nt"
                                else sub_info["path"]
                            )
                            if norm_key not in seen_paths:
                                seen_paths.add(norm_key)
                                items.append(sub_info)

        return {
            "default_dir": str(self.skills_dir),
            "scanned_dirs": [str(r) for r in roots],
            "active_skill_path": active_skill_path,
            "items": items,
        }

    def scan_all(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a unified overview of engines, models (including multimodal), and skills libraries."""
        self.ensure_default_directories()
        return {
            "engines": self.scan_engines(config),
            "models": self.scan_models(config),
            "skills": self.scan_skills(config),
        }
