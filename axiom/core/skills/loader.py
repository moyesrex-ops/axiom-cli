"""Skill loader — discover and load SKILL.md files from disk.

Skills are Markdown files that contain domain-specific knowledge,
patterns, and instructions that can be injected into the agent's
system prompt to enhance its capabilities for specific tasks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default skill directories (searched in order)
DEFAULT_SKILL_DIRS: list[Path] = [
    Path.home() / ".axiom" / "skills",          # User-level skills
    Path.home() / ".claude" / "skills",          # Claude Code skills (if installed)
    Path("memory") / "skills",                    # Project memory skills
]


@dataclass
class Skill:
    """A loaded skill with metadata and content."""

    name: str
    description: str = ""
    content: str = ""
    source_path: Path | None = None
    tags: list[str] = field(default_factory=list)
    token_estimate: int = 0

    def __post_init__(self) -> None:
        # Rough token estimate: ~4 chars per token
        self.token_estimate = len(self.content) // 4


class SkillLoader:
    """Discover and load SKILL.md files from configured directories.

    Skills are loaded lazily — directories are scanned on first access.
    Skills are cached in memory for the session lifetime.
    """

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self._dirs = list(DEFAULT_SKILL_DIRS)
        if extra_dirs:
            self._dirs.extend(extra_dirs)
        self._skills: dict[str, Skill] = {}
        self._loaded = False

    def load_all(self) -> int:
        """Scan all skill directories and load skills.

        Returns the number of skills loaded.
        """
        self._skills.clear()

        for skill_dir in self._dirs:
            if not skill_dir.exists() or not skill_dir.is_dir():
                continue
            self._scan_directory(skill_dir)

        self._loaded = True
        logger.info("Loaded %d skills from %d directories", len(self._skills), len(self._dirs))
        return len(self._skills)

    def _scan_directory(self, directory: Path) -> None:
        """Recursively scan a directory for SKILL.md files."""
        try:
            # Look for SKILL.md files (the standard name)
            for skill_file in directory.rglob("SKILL.md"):
                try:
                    skill = self._parse_skill_file(skill_file)
                    if skill and skill.name not in self._skills:
                        self._skills[skill.name] = skill
                except Exception as exc:
                    logger.debug("Failed to load skill %s: %s", skill_file, exc)
        except PermissionError:
            logger.debug("Permission denied scanning %s", directory)

    def _parse_skill_file(self, path: Path) -> Optional[Skill]:
        """Parse a SKILL.md file into a Skill object.

        Expected format (YAML-like frontmatter):
        ---
        name: skill-name
        description: What this skill does
        ---
        # Skill content in markdown...
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        if not text.strip():
            return None

        name = ""
        description = ""
        content = text
        tags: list[str] = []

        # Try to parse frontmatter (--- delimited)
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)",
            text,
            re.DOTALL,
        )
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            content = frontmatter_match.group(2).strip()

            # Simple YAML-like parsing (no pyyaml dependency needed)
            for line in frontmatter.split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    name = line[5:].strip().strip("\"'")
                elif line.startswith("description:"):
                    description = line[12:].strip().strip("\"'")
                elif line.startswith("tags:"):
                    # tags: [tag1, tag2] or tags: tag1, tag2
                    tag_str = line[5:].strip().strip("[]")
                    tags = [t.strip().strip("\"'") for t in tag_str.split(",") if t.strip()]

        # Fallback: derive name from directory or filename
        if not name:
            # Use parent directory name (e.g., skills/brainstorming/SKILL.md -> "brainstorming")
            parent = path.parent.name
            if parent and parent not in ("skills", "."):
                name = parent
            else:
                name = path.stem.lower()

        # Fallback: derive description from first line of content
        if not description and content:
            first_line = content.split("\n")[0].strip().lstrip("#").strip()
            if first_line and len(first_line) < 200:
                description = first_line

        return Skill(
            name=name,
            description=description,
            content=content,
            source_path=path,
            tags=tags,
        )

    # -- Public API ----------------------------------------------------------

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        if not self._loaded:
            self.load_all()
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """Return all loaded skills sorted by name."""
        if not self._loaded:
            self.load_all()
        return sorted(self._skills.values(), key=lambda s: s.name)

    def search(self, query: str) -> list[Skill]:
        """Search skills by name, description, or tags (simple substring match)."""
        if not self._loaded:
            self.load_all()

        query_lower = query.lower()
        results = []
        for skill in self._skills.values():
            if (
                query_lower in skill.name.lower()
                or query_lower in skill.description.lower()
                or any(query_lower in tag.lower() for tag in skill.tags)
            ):
                results.append(skill)
        return sorted(results, key=lambda s: s.name)

    @property
    def count(self) -> int:
        """Number of loaded skills."""
        if not self._loaded:
            self.load_all()
        return len(self._skills)

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens across all skills."""
        if not self._loaded:
            self.load_all()
        return sum(s.token_estimate for s in self._skills.values())
