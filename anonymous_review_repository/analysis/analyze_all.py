"""Run the nine result-analysis scripts in order."""

import sys
from pathlib import Path

from analyze_e1 import analyze as analyze_e1
from analyze_e2 import analyze as analyze_e2
from analyze_e3 import analyze as analyze_e3
from analyze_e4 import analyze as analyze_e4
from analyze_e5 import analyze as analyze_e5
from analyze_e6 import analyze as analyze_e6
from analyze_e7 import analyze as analyze_e7
from analyze_e8 import analyze as analyze_e8
from analyze_e9 import analyze as analyze_e9


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("analysis_output")

    analyses = [
        ("E1", analyze_e1),
        ("E2", analyze_e2),
        ("E3", analyze_e3),
        ("E4", analyze_e4),
        ("E5", analyze_e5),
        ("E6", analyze_e6),
        ("E7", analyze_e7),
        ("E8", analyze_e8),
        ("E9", analyze_e9),
    ]

    for name, function in analyses:
        print(f"Analyzing {name}", flush=True)
        function(results_dir, output_dir)

    print(f"Finished. Analysis files are in {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
