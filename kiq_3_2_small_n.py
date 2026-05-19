"""KIQ 3.2 — Small-N storytelling visuals for Name Tag sentiment.

The Name Tag corpus is small (21 posts, 125 comments after relevance filtering),
so distribution histograms are visually misleading. This script produces three
small-N-honest views suitable for a dashboard:

  1. Stacked horizontal bar — pos/neu/neg share per slice (the headline KPI).
  2. Strip plot              — one dot per item, jittered, colored by polarity.
                               Makes the sample size visible while showing
                               distribution shape.
  3. Quote wall              — top negative quotes + top positive(s) rendered
                               as a single PNG, ready to drop into the deck.

Inputs come from outputs/kiq_3_2/ (already LLM-scored), no re-scoring.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_2"
OUT_DIR = IN_DIR

COLORS = {"positive": "#2a9d8f", "neutral": "#adb5bd", "negative": "#e76f51"}


def stacked_bar(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "KIQ 3.2 — Name Tag sentiment: posts vs comments",
) -> None:
    def counts_and_shares(df: pd.DataFrame) -> dict[str, tuple[int, float]]:
        n = max(len(df), 1)
        out = {}
        for lab in COLORS:
            c = int((df["sentiment_label"] == lab).sum())
            out[lab] = (c, c / n * 100)
        return out

    rows = [
        ("Posts", len(posts), counts_and_shares(posts)),
        ("Comments", len(comments), counts_and_shares(comments)),
    ]
    labels = [f"{name}\n(n={n})" for name, n, _ in rows]
    pos = [r[2]["positive"][1] for r in rows]
    neu = [r[2]["neutral"][1] for r in rows]
    neg = [r[2]["negative"][1] for r in rows]
    pos_c = [r[2]["positive"][0] for r in rows]
    neu_c = [r[2]["neutral"][0] for r in rows]
    neg_c = [r[2]["negative"][0] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    y = np.arange(len(rows))
    ax.barh(y, pos, color=COLORS["positive"], label="positive")
    ax.barh(y, neu, left=pos, color=COLORS["neutral"], label="neutral")
    ax.barh(
        y,
        neg,
        left=[p + n for p, n in zip(pos, neu)],
        color=COLORS["negative"],
        label="negative",
    )
    for i, (p, ne, ng, pc, nec, ngc) in enumerate(zip(pos, neu, neg, pos_c, neu_c, neg_c)):
        if p >= 4:
            ax.text(p / 2, i, f"{p:.0f}%\n({pc})", ha="center", va="center", color="white", fontsize=10)
        if ne >= 4:
            ax.text(p + ne / 2, i, f"{ne:.0f}%\n({nec})", ha="center", va="center", color="white", fontsize=10)
        if ng >= 4:
            ax.text(
                p + ne + ng / 2, i,
                f"{ng:.0f}%\n({ngc})",
                ha="center", va="center", color="white", fontsize=11, fontweight="bold",
            )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of items")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, ncol=3, bbox_to_anchor=(1.0, -0.6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def strip_plot(posts: pd.DataFrame, comments: pd.DataFrame, out_path: Path) -> None:
    rng = np.random.default_rng(seed=42)
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=100)

    for y_base, df, label in [
        (1.0, posts, f"Posts (n={len(posts)})"),
        (0.0, comments, f"Comments (n={len(comments)})"),
    ]:
        if df.empty:
            continue
        x = df["sentiment_score"].astype(float).values
        jitter = rng.uniform(-0.18, 0.18, size=len(df))
        y = np.full_like(x, y_base) + jitter
        colors = [COLORS[lab] for lab in df["sentiment_label"]]
        ax.scatter(x, y, c=colors, s=55, alpha=0.75, edgecolors="white", linewidths=0.6)

    ax.axvline(0, color="black", lw=0.6, linestyle="--")
    ax.set_yticks([0.0, 1.0], [f"Comments (n={len(comments)})", f"Posts (n={len(posts)})"])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("LLM sentiment score (−1 = very negative, +1 = very positive)")
    ax.set_title("KIQ 3.2 — Every Name Tag item, one dot")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [mpatches.Patch(color=COLORS[lab], label=lab) for lab in ("positive", "neutral", "negative")]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def quote_wall(quotes_df: pd.DataFrame, out_path: Path, *, n_neg: int = 4, n_pos: int = 2) -> None:
    """Pick a curated set of representative quotes and render them as a slide-ready PNG."""
    neg = (
        quotes_df[quotes_df["polarity"] == "negative"]
        .sort_values("sentiment_score")  # most negative first
        .head(n_neg)
    )
    pos = (
        quotes_df[quotes_df["polarity"] == "positive"]
        .sort_values("sentiment_score", ascending=False)
        .head(n_pos)
    )
    picks = pd.concat([neg, pos], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 0.7 + 1.4 * len(picks)), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(picks))
    ax.axis("off")
    ax.set_title("KIQ 3.2 — Name Tag in users' own words", loc="left", fontsize=14, pad=14)

    for i, row in enumerate(picks.itertuples()):
        y = len(picks) - i - 1
        bar_color = COLORS["negative"] if row.polarity == "negative" else COLORS["positive"]
        # left color strip
        ax.add_patch(plt.Rectangle((0.0, y + 0.1), 0.012, 0.8, color=bar_color, clip_on=False))
        # meta line (chip)
        chip = f"{row.polarity.upper()}  •  score {row.sentiment_score:+.1f}  •  r/{row.subreddit}  •  {row.kind}  •  {row.created_utc}"
        ax.text(0.025, y + 0.78, chip, fontsize=8.5, color="#444")
        # quote text, wrapped
        text = str(row.text).strip().replace("\n", " ")
        wrapped = "\n".join(textwrap.wrap(text, width=110))
        if wrapped.count("\n") > 2:
            # cap at 3 lines + ellipsis
            lines = wrapped.split("\n")
            wrapped = "\n".join(lines[:3])
            if not wrapped.endswith("…"):
                wrapped += " …"
        ax.text(0.025, y + 0.42, f"“{wrapped}”", fontsize=10.2, color="#111", va="center")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    posts = pd.read_csv(IN_DIR / "kiq_3_2_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_3_2_scored_comments.csv")
    quotes = pd.read_csv(IN_DIR / "kiq_3_2_top_quotes.csv")

    posts = posts[posts["is_relevant"] == True].copy()
    comments = comments[comments["is_relevant"] == True].copy()
    print(f"Posts (relevant): {len(posts)} | Comments (relevant): {len(comments)}")

    stacked_bar(posts, comments, OUT_DIR / "kiq_3_2_stacked_bar.png")
    strip_plot(posts, comments, OUT_DIR / "kiq_3_2_strip.png")
    quote_wall(quotes, OUT_DIR / "kiq_3_2_quote_wall.png")

    print(f"Wrote: {OUT_DIR}/kiq_3_2_stacked_bar.png")
    print(f"Wrote: {OUT_DIR}/kiq_3_2_strip.png")
    print(f"Wrote: {OUT_DIR}/kiq_3_2_quote_wall.png")


if __name__ == "__main__":
    main()
