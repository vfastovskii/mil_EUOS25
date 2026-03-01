from __future__ import annotations

import sys
from pathlib import Path

# Ensure package-qualified imports (opt_attn_net_feb.*) resolve when running pytest
# from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _REPO_ROOT.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
