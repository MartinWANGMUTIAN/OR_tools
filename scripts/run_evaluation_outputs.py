#!/usr/bin/env python3
"""
Generate the standalone evaluation outputs expected under results/evaluation/.

This keeps the repository-level outputs aligned with README expectations by
running both whole-order and partial-order basket replay scripts.
"""

from pathlib import Path
import runpy


ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


def main() -> None:
    scripts = [
        ROOT_DIR / "src" / "evaluation" / "basket_simulation_whole.py",
        ROOT_DIR / "src" / "evaluation" / "basket_simulation_partial.py",
    ]
    for script in scripts:
        print(f"[run] {script}")
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
