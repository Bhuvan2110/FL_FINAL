"""Pytest configuration for root-level tests."""
import sys
from pathlib import Path

# Add backend directory to sys.path so `import app` resolves correctly
backend_path = str(Path(__file__).resolve().parent.parent / "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
