"""Regenerate all statistical tables and paper figures from the included results."""

from analyze_results import main as analyze_results
from make_figures import main as make_figures


def main():
    analyze_results()
    make_figures()


if __name__ == "__main__":
    main()
