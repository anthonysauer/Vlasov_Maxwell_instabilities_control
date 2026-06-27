#!/usr/bin/env python
"""Regenerate all outputs produced by the main notebook.

Usage
-----
python scripts/reproduce_all_figures.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vm_control.plotting import make_all_outputs


def main() -> None:
    root = ROOT
    outputs = make_all_outputs(root / "outputs" / "figures")
    print(f"Regenerated figures: {len(outputs['regenerated'])}")
    print(f"Submitted reference figures copied: {len(outputs['submitted_reference'])}")
    print(f"Output directory: {root / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()
