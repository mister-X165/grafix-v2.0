"""Root launcher: works from any cwd.

Usage:
  python train.py --data data/stellar_alexey_chunks.jsonl --steps 200
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.argv[0] = str(ROOT / "model" / "train.py")
runpy.run_path(str(ROOT / "model" / "train.py"), run_name="__main__")
