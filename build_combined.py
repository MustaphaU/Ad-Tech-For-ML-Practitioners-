#!/usr/bin/env python3
"""Regenerate combined.md from the individual chapter_*.md files.

combined.md is the input to build_pdf.sh; it is a verbatim concatenation of the
chapter files joined by a Markdown horizontal-rule separator. Editing the chapter
files and running this script keeps the two in sync (previously there was no
regenerator, so combined.md silently drifted from the chapters).

Edit the chapter files, then run this script to regenerate combined.md.
"""
import sys
from pathlib import Path

CHAPTERS = [
    "chapter_00_preface.md",
    "chapter_01_ecosystem.md",
    "chapter_02_auction_theory.md",
    "chapter_03_rtb.md",
    "chapter_04_ctr_prediction.md",
    "chapter_05_bid_optimization.md",
    "chapter_06_deep_learning_ctr.md",
    "chapter_07_rl_formulation.md",
    "chapter_08_rl_algorithms.md",
    "chapter_09_budget_pacing.md",
    "chapter_10_multiagent_auction_design.md",
    "chapter_11_production.md",
    "chapter_12_advanced_frontiers.md",
    "chapter_13_capstone.md",
]

SEP = "\n\n---\n\n"          # standard inter-chapter separator

def build(root: Path) -> str:
    texts = [(root / c).read_text() for c in CHAPTERS]
    seps = [SEP] * (len(CHAPTERS) - 1)
    out = texts[0]
    for text, sep in zip(texts[1:], seps):
        out += sep + text
    return out

def main() -> int:
    root = Path(__file__).resolve().parent
    check = "--check" in sys.argv
    new = build(root)
    target = root / "combined.md"
    if check:
        current = target.read_text() if target.exists() else ""
        if current == new:
            print("combined.md is in sync with chapters.")
            return 0
        print("combined.md is OUT OF SYNC with chapters. Run without --check to regenerate.")
        return 1
    target.write_text(new)
    print(f"Wrote {target} ({len(new)} bytes) from {len(CHAPTERS)} chapters.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
