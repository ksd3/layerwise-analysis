"""Regenerate every summary table from the released experiment CSV files."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "outputs" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "analysis" / "analyze_all.py"), str(ROOT / "data"), str(output)], check=True)
    subprocess.run([sys.executable, str(ROOT / "analysis" / "analyze_review_additions.py")], check=True)


if __name__ == "__main__":
    main()
