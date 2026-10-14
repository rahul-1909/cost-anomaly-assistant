"""Root wrapper for ask.py CLI assistant."""
import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "AI Intern Case Study"
sys.path.insert(0, str(pkg_dir))

from ask import main

if __name__ == "__main__":
    main()
