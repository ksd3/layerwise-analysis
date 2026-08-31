"""Run the full GPU experiment pipeline and its final statistical corrections."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"


def main():
    arguments = sys.argv[1:]
    subprocess.run([sys.executable, str(EXPERIMENTS / "run_all_experiments.py"), *arguments], check=True)
    if "--smoke-test" in arguments:
        return
    experiment_python = EXPERIMENTS / ".venv" / "bin" / "python"
    subprocess.run([str(experiment_python), str(EXPERIMENTS / "fix_statistics.py")], check=True)
    subprocess.run([str(experiment_python), str(EXPERIMENTS / "make_layerwise_physics_maps.py")], check=True)


if __name__ == "__main__":
    main()
