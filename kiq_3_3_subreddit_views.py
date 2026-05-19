"""KIQ 3.3 — Subreddit-level breakdowns for the WHERE panel.

  kiq_3_3_subreddit_polarity.{csv,png}
      Top-6 subreddits, horizontal stacked bar of pos/neu/neg share.
      Lets us say: "the negative frame holds across communities, not just r/privacy".

  kiq_3_3_subreddit_kind.{csv,png}
      Top-6 subreddits, grouped bar of post-count vs comment-count.
      Lets us say: "this is a comment-driven story, not a headline-driven one".
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_3"
OUT_DIR = IN_DIR

COLORS = {"positive": "#2a9d8f", "neutral": "#adb5bd", "negative": "#e76f51"}
POST_COLOR = "#1f3b73"
COMMENT_COLOR = "#5b9bd5"


def load_combined() -> pd.DataFrame:
    posts = pd.read_csv(IN_DIR / "kiq_3_3_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_3_3_scored_comments.csv")
    posts = posts[posts["is_relevant"] == True].assign(_kind="post")
    comments = comments[comments["is_relevant"] == True].assign(_kind="comment")
    cols = ["subreddit", "sentiment_label", "sentiment_score", "_kind"]
    return pd.concat([posts[cols], comments[cols]], ignore_index=True)


def subreddit_polarity(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    counts = df.groupby(["subreddit", "sentiment_label"]).size().unstack(fill_value=0)
    counts["n"] = counts.sum(axis=1)
    counts = counts.sort_values("n", ascending=False).head(top_n)
    for lab in ("positive", "neutral", "negative"):
        if lab not in counts.columns:
            counts[lab] = 0
        counts[f"pct_{lab}"] = counts[lab] / counts["n"] * 100
    return counts.reset_index()


def subreddit_kind(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    counts = df.groupby(["subreddit", "_kind"]).size().unstack(fill_value=0)
    counts["n"] = counts.sum(axis=1)
    counts = counts.sort_values("n", ascending=False).head(top_n)
    for col in ("post", "comment"):
        if col not in counts.columns:
            counts[col] = 0
    return counts.reset_index()


def plot_polarity(df: pd.DataFrame, out_path: Path) -> None:
    df = df.iloc[::-1]  # largest at top
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.5), dpi=100)
    y = np.arange(len(df))
    left = np.zeros(len(df))
    for lab in ("positive", "neutral", "negative"):
        vals = df[f"pct_{lab}"].values
        ax.barh(y, vals, left=left, color=COLORS[lab], label=lab)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(
                    left[i] + v / 2, y[i], f"{v:.0f}%",
                    ha="center", va="center", color="white", fontsize=9,
                    fontweight="bold" if lab == "negative" else "normal",
                )
        left = left + vals
    labels = [f"r/{s}\n(n={n})" for s, n in zip(df["subreddit"], df["n"])]
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of items")
    ax.set_title("KIQ 3.3 — Sentiment polarity by subreddit (top 6 by volume)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kind(df: pd.DataFrame, out_path: Path) -> None:
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 1.4), dpi=100)
    y = np.arange(len(df))
    h = 0.38
    posts_n = df["post"].values
    comments_n = df["comment"].values
    ax.barh(y + h / 2, comments_n, height=h, color=COMMENT_COLOR, label="comments")
    ax.barh(y - h / 2, posts_n, height=h, color=POST_COLOR, label="posts")
    xmax = max(comments_n.max() if len(comments_n) else 0, 1)
    for i, (p, c) in enumerate(zip(posts_n, comments_n)):
        if p > 0:
            ax.text(p + xmax * 0.01, y[i] - h / 2, f"{int(p)}", va="center", fontsize=9, color="#222")
        if c > 0:
            ax.text(c + xmax * 0.01, y[i] + h / 2, f"{int(c)}", va="center", fontsize=9, color="#222")
    labels = [f"r/{s}" for s in df["subreddit"]]
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Item count")
    ax.set_title("KIQ 3.3 — Posts vs comments by subreddit (top 6 by volume)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_combined()
    print(f"Combined relevant items: {len(df)}")

    pol = subreddit_polarity(df)
    pol.to_csv(OUT_DIR / "kiq_3_3_subreddit_polarity.csv", index=False)
    plot_polarity(pol, OUT_DIR / "kiq_3_3_subreddit_polarity.png")
    print("\nSubreddit x polarity:")
    print(pol[["subreddit", "n", "pct_positive", "pct_neutral", "pct_negative"]].round(1).to_string(index=False))

    kind = subreddit_kind(df)
    kind.to_csv(OUT_DIR / "kiq_3_3_subreddit_kind.csv", index=False)
    plot_kind(kind, OUT_DIR / "kiq_3_3_subreddit_kind.png")
    print("\nSubreddit x kind:")
    print(kind[["subreddit", "post", "comment", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
