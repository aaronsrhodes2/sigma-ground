"""No control characters in tracked source files.

Cause (windows-shell skill, 'a regex ... in a file written by a Python patch script inside a bash heredoc comes out wrong with NO error'): the Bash tool collapses a backslash pair in a
heredoc, so a `\b` in a patch script becomes `\b`, which Python turns into a BACKSPACE character inside a normal string. It compiles, runs and silently never matches (2026-10-16:
fetch_horizons_batch.py rejected a correct Horizons response because of one 0x08). Files with backslash escapes are written with the Write/Edit tools; this test is the net under that rule.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {0x09, 0x0A, 0x0D, 0x0C}            # tab, newline, carriage return, form feed


class TestNoControlCharacters(unittest.TestCase):
    def test_python_and_shell_sources_have_no_stray_control_characters(self):
        files = subprocess.run(["git", "ls-files", "*.py", "*.sh"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        bad = []
        for f in files:
            p = ROOT / f
            if not p.exists():
                continue
            data = p.read_bytes()
            hits = sorted({b for b in data if b < 0x20 and b not in ALLOWED})
            if hits:
                bad.append((f, [hex(h) for h in hits]))
        self.assertEqual(bad, [], "control characters in tracked sources (a backslash escape eaten by a heredoc?)")


if __name__ == "__main__":
    unittest.main()
