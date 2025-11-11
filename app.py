import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.web.bootstrap import run


if __name__ == "__main__":
    target = str(PROJECT_ROOT / "src" / "ui" / "app.py")
    run(target, '', [], flag_options={})


