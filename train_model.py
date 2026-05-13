"""Run training from repo root: python train_model.py"""
import subprocess
import sys
from pathlib import Path

PM = Path(__file__).resolve().parent / "predictive-maintenance"
script = PM / "train_model.py"
sys.exit(subprocess.run([sys.executable, str(script)], cwd=PM).returncode)
