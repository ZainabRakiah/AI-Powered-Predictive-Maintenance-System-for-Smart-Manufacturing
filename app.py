"""Run Flask app from repo root: python app.py"""
import os
import sys
from pathlib import Path

PM = Path(__file__).resolve().parent / "predictive-maintenance"
script = PM / "app.py"
os.chdir(PM)
os.execv(sys.executable, [sys.executable, str(script)])
