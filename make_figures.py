"""Regenerate every main-text and appendix figure used by the paper."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "outputs" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    marimo = Path(sys.executable).parent / "marimo"
    subprocess.run([str(marimo), "export", "html", str(ROOT / "figures" / "main_figures_notebook.py"), "--no-include-code", "-o", str(output / "main_figures.html"), "-f"], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "plot_physics_alignment.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "05_normalized_depth_atlas.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "06_cross_layer_atlas.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "05_geometry_vs_physics.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "06_final_transfer_calibration.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "figures" / "07_common_space_retrieval.py")], check=True)


if __name__ == "__main__":
    main()
