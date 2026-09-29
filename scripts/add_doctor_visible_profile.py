"""Deprecated guard: dialogue text must not be rewritten as a patient chart.

Doctor-R1 trains the doctor from the dialogue history. This project does not
have a verified pre-visit EHR field in its source cases, so profile synthesis
from conversation turns is disabled.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.parse_args()
    parser.error(
        "Disabled: no verified pre-visit patient-chart source is available. "
        "Use only messages already present in the original consultation dialogue."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
