"""Self-repair & introspection tool -- lets Axiom inspect and fix its own codebase.

GOD MODE: The agent can read its own source code, edit it, diagnose
runtime errors, hot-reload modules, and evolve autonomously.

This is the key differentiator: Axiom can fix itself without human
intervention.  When the agent detects a bug in its own behaviour it
can:

1.  Read the faulty module with ``introspect``
2.  Identify the bug (using LLM reasoning)
3.  Apply a surgical edit with ``self_edit``
4.  Hot-reload the patched module with ``hot_reload``
5.  Verify the fix by exercising the patched path

Safety: edits are logged to ``memory/self_repairs/`` so the user can
audit and revert if needed.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)

# Root of the axiom package on disk
_AXIOM_ROOT = Path(__file__).resolve().parent.parent.parent  # axiom/
_PROJECT_ROOT = _AXIOM_ROOT.parent  # axiom-cli/


class SelfRepairTool(AxiomTool):
    """Introspect and repair Axiom's own codebase at runtime.

    Actions
    -------
    introspect  : Read any Axiom source file by module path or relative path.
    self_edit   : Apply a surgical string replacement to an Axiom source file.
    hot_reload  : Reimport a module so the fix takes effect without restart.
    diagnose    : Capture the last runtime error with full traceback.
    codebase_map: Return the full directory tree of the Axiom project.
    get_config  : Show current runtime state (model, tools, memory stats).
    """

    name = "self_repair"
    description = (
        "Inspect and repair Axiom's own source code at runtime. "
        "Actions: introspect (read source), self_edit (patch code), "
        "hot_reload (reimport module), diagnose (get last error), "
        "codebase_map (directory tree), get_config (runtime state). "
        "Use this when you detect a bug in yourself or need to add "
        "new capabilities dynamically."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "One of: introspect, self_edit, hot_reload, diagnose, "
                    "codebase_map, get_config"
                ),
                "enum": [
                    "introspect",
                    "self_edit",
                    "hot_reload",
                    "diagnose",
                    "codebase_map",
                    "get_config",
                ],
            },
            "target": {
                "type": "string",
                "description": (
                    "For introspect/self_edit/hot_reload: module path "
                    "(e.g. 'axiom.cli.renderer') or relative file path "
                    "(e.g. 'axiom/cli/renderer.py'). "
                    "For diagnose: optional error context."
                ),
            },
            "old_text": {
                "type": "string",
                "description": "For self_edit: exact text to replace.",
            },
            "new_text": {
                "type": "string",
                "description": "For self_edit: replacement text.",
            },
        },
        "required": ["action"],
    }
    risk_level = "high"  # Always require approval for self-modification

    # Shared state for error capture
    _last_error: str = ""
    _last_traceback: str = ""
    _repair_log: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        target = kwargs.get("target", "")

        if action == "introspect":
            return self._introspect(target)
        elif action == "self_edit":
            old_text = kwargs.get("old_text", "")
            new_text = kwargs.get("new_text", "")
            return self._self_edit(target, old_text, new_text)
        elif action == "hot_reload":
            return self._hot_reload(target)
        elif action == "diagnose":
            return self._diagnose(target)
        elif action == "codebase_map":
            return self._codebase_map()
        elif action == "get_config":
            return self._get_config()
        else:
            raise ToolError(
                f"Unknown action: {action}. "
                "Use: introspect, self_edit, hot_reload, diagnose, "
                "codebase_map, get_config",
                tool_name=self.name,
            )

    # ── Actions ──────────────────────────────────────────────────

    def _introspect(self, target: str) -> str:
        """Read an Axiom source file by module path or relative path."""
        path = self._resolve_path(target)
        if not path.exists():
            return f"Error: file not found: {path}"
        if not self._is_safe_path(path):
            return f"Error: path outside Axiom project: {path}"

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")
            # Add line numbers
            numbered = "\n".join(
                f"{i+1:>4} | {line}" for i, line in enumerate(lines)
            )
            return (
                f"File: {path.relative_to(_PROJECT_ROOT)}\n"
                f"Lines: {len(lines)}\n"
                f"Size: {len(content)} bytes\n\n"
                f"{numbered}"
            )
        except Exception as exc:
            return f"Error reading {path}: {exc}"

    def _self_edit(self, target: str, old_text: str, new_text: str) -> str:
        """Apply a surgical string replacement to an Axiom source file."""
        if not old_text:
            return "Error: old_text is required for self_edit"
        if not new_text:
            return "Error: new_text is required for self_edit"
        if old_text == new_text:
            return "Error: old_text and new_text are identical"

        path = self._resolve_path(target)
        if not path.exists():
            return f"Error: file not found: {path}"
        if not self._is_safe_path(path):
            return f"Error: path outside Axiom project: {path}"

        try:
            content = path.read_text(encoding="utf-8")

            # Verify old_text exists (exactly once for safety)
            count = content.count(old_text)
            if count == 0:
                return (
                    f"Error: old_text not found in {path.name}. "
                    "Make sure you're using the exact text from introspect."
                )
            if count > 1:
                return (
                    f"Warning: old_text appears {count} times in {path.name}. "
                    "Provide more context to make the match unique. "
                    "Use introspect to see the file first."
                )

            # Apply the edit
            new_content = content.replace(old_text, new_text, 1)

            # Backup original (in memory for immediate revert)
            backup_key = f"{path.name}_{int(time.time())}"

            # Write the patched file
            path.write_text(new_content, encoding="utf-8")

            # Log the repair
            repair_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file": str(path.relative_to(_PROJECT_ROOT)),
                "old_text_preview": old_text[:100],
                "new_text_preview": new_text[:100],
                "backup_key": backup_key,
            }
            self._repair_log.append(repair_entry)

            # Save repair log to memory
            self._save_repair_log(repair_entry, content)

            rel = path.relative_to(_PROJECT_ROOT)
            return (
                f"✓ Successfully patched {rel}\n"
                f"  Replaced {len(old_text)} chars with {len(new_text)} chars\n"
                f"  Backup saved as: {backup_key}\n"
                f"  Use hot_reload to apply changes without restart."
            )

        except Exception as exc:
            return f"Error editing {path}: {exc}"

    def _hot_reload(self, target: str) -> str:
        """Reimport a module so edits take effect without restart."""
        module_name = self._to_module_name(target)
        if not module_name:
            return f"Error: cannot determine module name for: {target}"

        # Find and reload the module
        if module_name in sys.modules:
            try:
                mod = sys.modules[module_name]
                importlib.reload(mod)
                return (
                    f"✓ Hot-reloaded module: {module_name}\n"
                    f"  Changes are now active in the current session."
                )
            except Exception as exc:
                tb = traceback.format_exc()
                return (
                    f"Error reloading {module_name}: {exc}\n"
                    f"Traceback:\n{tb}\n"
                    f"The edit may have introduced a syntax error. "
                    f"Use introspect to review the file and self_edit to fix."
                )
        else:
            return (
                f"Module {module_name} is not currently loaded. "
                f"Changes will take effect on next import / restart."
            )

    def _diagnose(self, context: str = "") -> str:
        """Return the last captured runtime error with full traceback."""
        parts = []

        if self._last_error:
            parts.append(f"Last Error: {self._last_error}")
        if self._last_traceback:
            parts.append(f"\nFull Traceback:\n{self._last_traceback}")
        if self._repair_log:
            parts.append(f"\nRecent Repairs ({len(self._repair_log)}):")
            for entry in self._repair_log[-5:]:
                parts.append(
                    f"  - [{entry['timestamp']}] {entry['file']}: "
                    f"{entry['old_text_preview'][:50]}..."
                )

        # Add runtime diagnostics
        parts.append("\n── Runtime Diagnostics ──")
        parts.append(f"Python: {sys.version}")
        parts.append(f"Platform: {sys.platform}")
        parts.append(f"CWD: {os.getcwd()}")
        parts.append(f"Axiom root: {_AXIOM_ROOT}")
        parts.append(f"Loaded modules (axiom.*): {self._count_axiom_modules()}")

        if context:
            parts.append(f"\nAdditional context: {context}")

        if not parts:
            return "No errors captured. System is healthy."

        return "\n".join(parts)

    def _codebase_map(self) -> str:
        """Return the full directory tree of the Axiom project."""
        lines = []
        for root, dirs, files in os.walk(_AXIOM_ROOT):
            # Skip __pycache__ and hidden dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith("__") and not d.startswith(".")
            ]
            level = Path(root).relative_to(_AXIOM_ROOT)
            indent = "  " * len(level.parts)
            dir_name = Path(root).name
            lines.append(f"{indent}{dir_name}/")
            for f in sorted(files):
                if f.endswith((".py", ".md", ".yaml", ".yml", ".json", ".toml")):
                    fpath = Path(root) / f
                    size = fpath.stat().st_size if fpath.exists() else 0
                    size_kb = f"{size/1024:.1f}KB"
                    lines.append(f"{indent}  {f} ({size_kb})")

        return (
            f"Axiom Project Structure ({_PROJECT_ROOT})\n"
            f"{'=' * 50}\n"
            + "\n".join(lines)
        )

    def _get_config(self) -> str:
        """Show current runtime state."""
        parts = [
            "── Axiom Runtime Configuration ──",
            f"Project Root: {_PROJECT_ROOT}",
            f"Package Root: {_AXIOM_ROOT}",
            f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            f"Platform: {sys.platform}",
            f"CWD: {os.getcwd()}",
            "",
            "Loaded Axiom Modules:",
        ]

        axiom_mods = sorted(
            name for name in sys.modules
            if name.startswith("axiom.") and sys.modules[name] is not None
        )
        for mod_name in axiom_mods:
            mod = sys.modules[mod_name]
            file_path = getattr(mod, "__file__", "built-in")
            parts.append(f"  {mod_name}: {file_path}")

        parts.append(f"\nTotal axiom modules loaded: {len(axiom_mods)}")
        parts.append(f"Self-repairs this session: {len(self._repair_log)}")

        return "\n".join(parts)

    # ── Class-level error capture (called from app.py) ────────────

    @classmethod
    def capture_error(cls, error: Exception, tb: str = "") -> None:
        """Capture a runtime error for the diagnose action."""
        cls._last_error = f"{type(error).__name__}: {error}"
        cls._last_traceback = tb or traceback.format_exc()

    @classmethod
    def get_last_error(cls) -> str:
        """Return the last captured error string."""
        return cls._last_error

    # ── Helpers ───────────────────────────────────────────────────

    def _resolve_path(self, target: str) -> Path:
        """Convert a module path or relative path to an absolute file path."""
        if not target:
            return _AXIOM_ROOT / "__init__.py"

        # If it's a dotted module path (axiom.cli.renderer)
        if "." in target and not target.endswith(".py"):
            parts = target.replace(".", os.sep) + ".py"
            candidate = _PROJECT_ROOT / parts
            if candidate.exists():
                return candidate

        # If it's a relative path (axiom/cli/renderer.py)
        candidate = _PROJECT_ROOT / target
        if candidate.exists():
            return candidate

        # Try as just a filename within axiom/
        for root, _, files in os.walk(_AXIOM_ROOT):
            for f in files:
                if f == target or f == target + ".py":
                    return Path(root) / f

        return _PROJECT_ROOT / target

    def _is_safe_path(self, path: Path) -> bool:
        """Ensure the path is within the Axiom project directory."""
        try:
            path.resolve().relative_to(_PROJECT_ROOT.resolve())
            return True
        except ValueError:
            return False

    def _to_module_name(self, target: str) -> str:
        """Convert a target to a Python module name."""
        if not target:
            return ""

        # Already a module name
        if "." in target and not target.endswith(".py"):
            return target

        # Convert file path to module name
        if target.endswith(".py"):
            target = target[:-3]
        return target.replace(os.sep, ".").replace("/", ".")

    def _count_axiom_modules(self) -> int:
        """Count loaded axiom.* modules."""
        return sum(
            1 for name in sys.modules
            if name.startswith("axiom.") and sys.modules[name] is not None
        )

    def _save_repair_log(
        self, entry: dict[str, Any], original_content: str
    ) -> None:
        """Save repair log entry and backup to memory/self_repairs/."""
        repair_dir = _PROJECT_ROOT / "memory" / "self_repairs"
        repair_dir.mkdir(parents=True, exist_ok=True)

        # Save the log entry
        log_file = repair_dir / "repair_log.md"
        log_line = (
            f"\n## {entry['timestamp']}\n"
            f"- **File**: `{entry['file']}`\n"
            f"- **Old**: `{entry['old_text_preview']}`\n"
            f"- **New**: `{entry['new_text_preview']}`\n"
        )
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
        except Exception:
            pass

        # Save backup of original file
        backup_file = repair_dir / f"{entry['backup_key']}.bak"
        try:
            backup_file.write_text(original_content, encoding="utf-8")
        except Exception:
            pass
