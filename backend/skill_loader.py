"""Markdown-only character skill loader with YAML frontmatter extraction."""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CORE_PRIORITY_ORDER: tuple[str, ...] = (
    "skill.md",
    "prompt.md",
    "profile.md",
    "personality.md",
    "interaction.md",
    "relations.md",
    "conflicts.md",
    "memory.md",
)

META_DOC_ORDER: tuple[str, ...] = (
    "readme.md",
    "quality-report.md",
    "test-report.md",
    "roleplay-test-report.md",
    "changelog.md",
)

CORE_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CORE_PRIORITY_ORDER)}
META_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(META_DOC_ORDER)}

LARGE_PROMPT_CHARS_THRESHOLD = 16_000
LARGE_PROMPT_TOKENS_THRESHOLD = 8_000

_TRUNCATION_SUFFIX = "\n\n[Truncated due to max_prompt_chars limit]"
_FENCE_START_RE = re.compile(r"^---\s*$")
_FENCE_END_RE = re.compile(r"^(?:---|\.\.\.)\s*$")
_YAML_KEY_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*:(?:\s+(.*)|\s*$)")
_BLOCK_HEADER_RE = re.compile(r"^([|>])([+-]?)(\d*)(?:\s+#.*)?$")
_DOUBLE_UNESCAPE_RE = re.compile(r'\\(["\\nrt])')
_UNESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class SkillLoadError(Exception):
    """Structured error raised when loading a character skill fails."""

    def __init__(
        self,
        code: str,
        message: str,
        detail: str = "",
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.http_status = http_status

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(slots=True)
class SkillLoadResult:
    """Loaded character skill metadata, assembled prompt, and diagnostics."""

    character_name: str
    description: str
    source_path: str
    source_type: Literal["file", "directory"]
    system_prompt: str
    loaded_files: list[str] = field(default_factory=list)
    ignored_files: list[str] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)
    total_chars: int = 0
    estimated_tokens: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return max(1, math.ceil(non_ascii * 1.0 + ascii_chars / 4.0))


def _canonical_sort_key(rel_path: str) -> tuple[int, int, str, str]:
    name_lower = Path(rel_path).name.lower()
    if name_lower in CORE_INDEX:
        return (0, CORE_INDEX[name_lower], "", rel_path.lower())
    if name_lower in META_INDEX:
        return (2, META_INDEX[name_lower], "", rel_path.lower())
    return (1, 0, name_lower, rel_path.lower())


def _parse_block_scalar(
    fm_lines: list[str],
    start_idx: int,
    style: str,
    chomp: str,
    indent_hint: str,
) -> tuple[str, int]:
    """Parse a YAML | or > indented multiline block scalar."""
    collected: list[str] = []
    idx = start_idx
    while idx < len(fm_lines):
        raw_line = fm_lines[idx]
        if raw_line.strip() == "":
            collected.append("")
            idx += 1
            continue
        if raw_line[0] in (" ", "\t"):
            collected.append(raw_line)
            idx += 1
            continue
        break

    # Determine base indentation
    base_indent: int | None = int(indent_hint) if indent_hint else None
    if base_indent is None:
        for line in collected:
            if line.strip():
                base_indent = len(line) - len(line.lstrip(" "))
                break
    if base_indent is None:
        base_indent = 0

    stripped_lines: list[str] = []
    for line in collected:
        if line == "":
            stripped_lines.append("")
        elif line.startswith(" " * base_indent):
            stripped_lines.append(line[base_indent:])
        else:
            stripped_lines.append(line.lstrip(" \t"))

    # Handle chomping (- strips trailing blanks, default clip keeps 1 newline, + keeps all)
    if chomp == "-" or chomp == "":
        while stripped_lines and stripped_lines[-1] == "":
            stripped_lines.pop()

    if not stripped_lines:
        return "", idx

    if style == "|":
        text = "\n".join(stripped_lines)
        if chomp == "":
            text += "\n"
        return text, idx

    # Folded style '>'
    paragraphs: list[str] = []
    current: list[str] = []
    for line in stripped_lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append("")
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    text = "\n".join(paragraphs)
    if chomp == "":
        text += "\n"
    return text, idx


def _unquote_scalar(raw: str) -> str:
    val = raw.strip()
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        inner = val[1:-1]
        return _DOUBLE_UNESCAPE_RE.sub(lambda m: _UNESCAPE_MAP[m.group(1)], inner)
    if len(val) >= 2 and val[0] == "'" and val[-1] == "'":
        return val[1:-1].replace("''", "'")
    return val


def _parse_simple_yaml(fm_lines: list[str]) -> dict[str, str]:
    """Parse top-level YAML key/value pairs including block scalars and quoted/JSON values."""
    result: dict[str, str] = {}
    idx = 0
    n = len(fm_lines)
    while idx < n:
        line = fm_lines[idx]
        if not line.strip() or line.lstrip().startswith("#"):
            idx += 1
            continue
        match = _YAML_KEY_RE.match(line)
        if not match:
            idx += 1
            continue
        key = match.group(1)
        rest = (match.group(2) or "").strip()
        block_match = _BLOCK_HEADER_RE.match(rest)
        if block_match:
            style, chomp, indent_hint = block_match.groups()
            val, idx = _parse_block_scalar(fm_lines, idx + 1, style, chomp, indent_hint)
            result[key] = val.strip() if style in ("|", ">") else val
            continue
        if rest == "":
            # Nested mapping/list under key: skip indented lines gracefully
            idx += 1
            while idx < n and (fm_lines[idx].strip() == "" or fm_lines[idx][0] in (" ", "\t")):
                idx += 1
            result[key] = ""
            continue
        result[key] = _unquote_scalar(rest)
        idx += 1
    return result


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Strip only leading YAML frontmatter, preserving all internal '---' horizontal rules."""
    lines = text.split("\n")
    if not lines or not _FENCE_START_RE.match(lines[0]):
        return {}, text

    close_idx: int | None = None
    for idx in range(1, len(lines)):
        if _FENCE_END_RE.match(lines[idx]):
            close_idx = idx
            break

    if close_idx is None:
        return {}, text

    fm_lines = lines[1:close_idx]
    is_empty = all(not ln.strip() for ln in fm_lines)
    has_top_level_key = any(_YAML_KEY_RE.match(ln) for ln in fm_lines)
    if not (is_empty or has_top_level_key):
        return {}, text

    metadata = _parse_simple_yaml(fm_lines)
    body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")
    return metadata, body


def _read_markdown_file(file_path: Path) -> tuple[dict[str, str], str]:
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        raise SkillLoadError(
            code="SKILL_ACCESS_ERROR",
            message=f"Failed to read markdown file: {file_path.name}",
            detail=str(exc),
            http_status=400,
        ) from exc

    try:
        text = raw_bytes.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise SkillLoadError(
            code="SKILL_ENCODING_ERROR",
            message=f"Markdown file is not valid UTF-8: {file_path.name}",
            detail=str(exc),
            http_status=400,
        ) from exc

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return _split_frontmatter(normalized)


def _apply_size_controls(
    system_prompt: str,
    max_prompt_chars: int,
    warnings: list[str],
) -> tuple[str, int, int]:
    if max_prompt_chars > 0 and len(system_prompt) > max_prompt_chars:
        original_len = len(system_prompt)
        if max_prompt_chars > len(_TRUNCATION_SUFFIX):
            keep_len = max_prompt_chars - len(_TRUNCATION_SUFFIX)
            truncated = system_prompt[:keep_len].rstrip() + _TRUNCATION_SUFFIX
        else:
            truncated = system_prompt[:max_prompt_chars]
        system_prompt = truncated
        warnings.append(
            f"System prompt truncated from {original_len} to {len(system_prompt)} chars "
            f"(max_prompt_chars={max_prompt_chars})."
        )

    total_chars = len(system_prompt)
    estimated_tokens = _estimate_tokens(system_prompt)
    if (
        total_chars > LARGE_PROMPT_CHARS_THRESHOLD
        or estimated_tokens > LARGE_PROMPT_TOKENS_THRESHOLD
    ):
        warnings.append(
            f"Large skill prompt ({total_chars} chars, ~{estimated_tokens} est. tokens); "
            "ensure llama.ctx_size is sufficient or enable exclude_meta_docs / max_prompt_chars."
        )
    return system_prompt, total_chars, estimated_tokens


class SkillLoader:
    """Loader for single .md character skills or .md-only skill directories."""

    @classmethod
    def _scan_directory(cls, root: Path) -> tuple[list[str], list[str]]:
        try:
            entries = list(root.iterdir())
        except OSError as exc:
            raise SkillLoadError(
                code="SKILL_ACCESS_ERROR",
                message="Failed to list skill directory.",
                detail=str(exc),
                http_status=400,
            ) from exc

        if not entries:
            raise SkillLoadError(
                code="SKILL_EMPTY_DIRECTORY",
                message="Skill directory is empty.",
                detail=str(root),
                http_status=400,
            )

        top_md: list[str] = []
        top_ignored: list[str] = []
        top_subdirs: list[Path] = []

        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_file():
                    if entry.suffix.lower() == ".md":
                        top_md.append(entry.name)
                    else:
                        top_ignored.append(entry.name)
                elif entry.is_dir():
                    top_subdirs.append(entry)
            except OSError as exc:
                raise SkillLoadError(
                    code="SKILL_ACCESS_ERROR",
                    message=f"Failed to inspect directory entry: {entry.name}",
                    detail=str(exc),
                    http_status=400,
                ) from exc

        if top_md:
            return top_md, sorted(top_ignored, key=str.lower)

        nested_md: list[str] = []
        nested_ignored: list[str] = []

        def _on_walk_error(exc: OSError) -> None:
            raise SkillLoadError(
                code="SKILL_ACCESS_ERROR",
                message="Failed while traversing skill subdirectories.",
                detail=str(exc),
                http_status=400,
            ) from exc

        for subdir in sorted(top_subdirs, key=lambda p: p.name.lower()):
            for dirpath, dirnames, filenames in os.walk(
                subdir, topdown=True, followlinks=False, onerror=_on_walk_error
            ):
                dirnames[:] = sorted(
                    (d for d in dirnames if not d.startswith(".")),
                    key=str.lower,
                )
                for fname in sorted(filenames, key=str.lower):
                    if fname.startswith("."):
                        continue
                    fpath = Path(dirpath) / fname
                    rel = fpath.relative_to(root).as_posix()
                    if fpath.suffix.lower() == ".md":
                        nested_md.append(rel)
                    else:
                        nested_ignored.append(rel)

        ignored_files = sorted(top_ignored + nested_ignored, key=str.lower)
        if not nested_md:
            detail_msg = (
                f"Ignored non-markdown files: {', '.join(ignored_files)}"
                if ignored_files
                else "No readable .md files found (dot-entries are skipped)."
            )
            raise SkillLoadError(
                code="SKILL_NO_MARKDOWN_FILES",
                message="Directory contains no markdown (.md) files.",
                detail=detail_msg,
                http_status=400,
            )

        return nested_md, ignored_files

    @classmethod
    def load(
        cls,
        path: str | Path,
        exclude_meta_docs: bool = False,
        max_prompt_chars: int = 0,
    ) -> SkillLoadResult:
        """Load a single .md file or a directory containing .md character skill files."""
        if not isinstance(exclude_meta_docs, bool):
            raise SkillLoadError(
                code="SKILL_INVALID_ARGUMENT",
                message="Argument 'exclude_meta_docs' must be a boolean.",
                http_status=400,
            )
        if isinstance(max_prompt_chars, bool) or not isinstance(max_prompt_chars, int) or max_prompt_chars < 0:
            raise SkillLoadError(
                code="SKILL_INVALID_ARGUMENT",
                message="Argument 'max_prompt_chars' must be an integer >= 0.",
                http_status=400,
            )

        if path is None:
            raise SkillLoadError(
                code="SKILL_PATH_EMPTY",
                message="Skill path must not be empty.",
                http_status=400,
            )
        if isinstance(path, str):
            if not path.strip():
                raise SkillLoadError(
                    code="SKILL_PATH_EMPTY",
                    message="Skill path must not be empty.",
                    http_status=400,
                )
            raw_path = Path(path.strip())
        elif isinstance(path, Path):
            raw_path = path
        else:
            raise SkillLoadError(
                code="SKILL_INVALID_ARGUMENT",
                message="Skill path must be a str or pathlib.Path.",
                detail=f"Got {type(path).__name__}.",
                http_status=400,
            )

        try:
            expanded = raw_path.expanduser()
            exists = expanded.exists()
            resolved = expanded.resolve() if exists else expanded
        except OSError as exc:
            raise SkillLoadError(
                code="SKILL_ACCESS_ERROR",
                message="Failed to resolve skill path.",
                detail=str(exc),
                http_status=400,
            ) from exc

        if not exists:
            raise SkillLoadError(
                code="SKILL_PATH_NOT_FOUND",
                message="Skill path does not exist.",
                detail=str(raw_path),
                http_status=400,
            )

        source_path = str(resolved)
        warnings: list[str] = []

        try:
            is_file = resolved.is_file()
            is_dir = resolved.is_dir()
        except OSError as exc:
            raise SkillLoadError(
                code="SKILL_ACCESS_ERROR",
                message="Failed to inspect skill path.",
                detail=str(exc),
                http_status=400,
            ) from exc

        if is_file:
            if resolved.suffix.lower() != ".md":
                raise SkillLoadError(
                    code="SKILL_UNSUPPORTED_FILE",
                    message="Only markdown (.md) skill files are supported.",
                    detail=resolved.name,
                    http_status=400,
                )
            metadata, body = _read_markdown_file(resolved)
            body_stripped = body.strip()
            if not body_stripped:
                raise SkillLoadError(
                    code="SKILL_EMPTY_CONTENT",
                    message="Skill markdown file has no prompt content.",
                    detail=resolved.name,
                    http_status=400,
                )
            character_name = metadata.get("name", "").strip() or resolved.stem
            description = metadata.get("description", "").strip()
            system_prompt, total_chars, estimated_tokens = _apply_size_controls(
                body_stripped, max_prompt_chars, warnings
            )
            return SkillLoadResult(
                character_name=character_name,
                description=description,
                source_path=source_path,
                source_type="file",
                system_prompt=system_prompt,
                loaded_files=[resolved.name],
                ignored_files=[],
                excluded_files=[],
                total_chars=total_chars,
                estimated_tokens=estimated_tokens,
                warnings=warnings,
            )

        if not is_dir:
            raise SkillLoadError(
                code="SKILL_ACCESS_ERROR",
                message="Skill path must be a regular file or directory.",
                detail=source_path,
                http_status=400,
            )

        md_candidates, ignored_files = cls._scan_directory(resolved)
        sorted_candidates = sorted(md_candidates, key=_canonical_sort_key)

        excluded_files: list[str] = []
        files_to_read: list[str] = []
        for rel in sorted_candidates:
            if exclude_meta_docs and Path(rel).name.lower() in META_INDEX:
                excluded_files.append(rel)
            else:
                files_to_read.append(rel)

        if not files_to_read:
            raise SkillLoadError(
                code="SKILL_EMPTY_CONTENT",
                message="No markdown files remain after excluding meta docs.",
                detail=f"Excluded: {', '.join(excluded_files)}",
                http_status=400,
            )

        loaded_files: list[str] = []
        sections: list[str] = []
        first_name = ""
        first_description = ""

        for rel in files_to_read:
            file_path = resolved / rel
            metadata, body = _read_markdown_file(file_path)
            loaded_files.append(rel)

            if not first_name and metadata.get("name", "").strip():
                first_name = metadata["name"].strip()
            if not first_description and metadata.get("description", "").strip():
                first_description = metadata["description"].strip()

            body_stripped = body.strip()
            if not body_stripped:
                warnings.append(f"Skipped empty markdown file: {rel}")
            else:
                sections.append(f"<!-- module: {rel} -->\n{body_stripped}")

        if not sections:
            raise SkillLoadError(
                code="SKILL_EMPTY_CONTENT",
                message="All markdown files in skill directory are empty.",
                detail=source_path,
                http_status=400,
            )

        combined_prompt = "\n\n".join(sections)
        system_prompt, total_chars, estimated_tokens = _apply_size_controls(
            combined_prompt, max_prompt_chars, warnings
        )
        character_name = first_name or resolved.name
        description = first_description

        return SkillLoadResult(
            character_name=character_name,
            description=description,
            source_path=source_path,
            source_type="directory",
            system_prompt=system_prompt,
            loaded_files=loaded_files,
            ignored_files=ignored_files,
            excluded_files=excluded_files,
            total_chars=total_chars,
            estimated_tokens=estimated_tokens,
            warnings=warnings,
        )
