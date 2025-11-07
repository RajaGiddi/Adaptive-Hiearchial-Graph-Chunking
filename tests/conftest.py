"""Pytest configuration to ensure local package imports resolve.

Adds the repository root to sys.path so tests can `import ahgc` without
installing the package. This mirrors the approach used in scripts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
