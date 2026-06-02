import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"

for path in (str(ROOT_DIR), str(BACKEND_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
