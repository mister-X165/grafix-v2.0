"""Root launcher: python import_stellar.py"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.argv[0] = str(ROOT / "data" / "import_stellar.py")
runpy.run_path(str(ROOT / "data" / "import_stellar.py"), run_name="__main__")
