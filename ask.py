"""
ask.py -- Stretch goal CLI for plain-English questions.

Usage:
    python ask.py "why did route Delhi-Jaipur get pricier in March?"
    python ask.py   # interactive mode
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import main

if __name__ == "__main__":
    main()
