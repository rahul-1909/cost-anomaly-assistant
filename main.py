"""Root wrapper for running main pipeline."""
import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "AI Intern Case Study"
sys.path.insert(0, str(pkg_dir))

from main import main

if __name__ == "__main__":
    main()
