"""KIQ 3.3 — Stacked horizontal sentiment bar, posts vs comments.

Same visual style as kiq_3_2_stacked_bar.png — % labels inside segments, sample
size in the row label — so the dashboard reads consistently across tabs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kiq_3_2_small_n import stacked_bar

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_3"
OUT_DIR = IN_DIR


def main() -> None:
    posts = pd.read_csv(IN_DIR / "kiq_3_3_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_3_3_scored_comments.csv")
    print(f"Posts: {len(posts)} | Comments: {len(comments)}")

    out = OUT_DIR / "kiq_3_3_stacked_bar.png"
    stacked_bar(
        posts,
        comments,
        out,
        title="KIQ 3.3 — Privacy content sentiment: posts vs comments",
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
