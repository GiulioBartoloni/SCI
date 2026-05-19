"""KIQ 3.3 — Post-comment relationship views.

Three competing renderings of the same fact: comments are reactions to specific
posts, and their tone can disagree with the parent post's tone.

  A. Force-directed network (networkx)        -> *_network.png
  B. Post-column beeswarm (sorted)            -> *_beeswarm.png
  C. Post-vs-mean-comment scatter             -> *_scatter.png

All three read the cached scored CSVs; no LLM re-scoring required.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_3"
OUT_DIR = IN_DIR

COLORS = {"positive": "#2a9d8f", "neutral": "#adb5bd", "negative": "#e76f51"}
LEGEND_PATCHES = [mpatches.Patch(color=COLORS[lab], label=lab) for lab in ("positive", "neutral", "negative")]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(IN_DIR / "kiq_3_3_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_3_3_scored_comments.csv")
    # Keep only comments whose parent post is in the scored set.
    comments = comments[comments["post_id"].isin(posts["post_id"])].copy()
    return posts, comments


# ---------- A. Force-directed network ---------------------------------------

def network(posts: pd.DataFrame, comments: pd.DataFrame, out_path: Path) -> None:
    parents = comments["post_id"].unique()
    posts_in_graph = posts[posts["post_id"].isin(parents)].copy()

    g = nx.Graph()
    for _, p in posts_in_graph.iterrows():
        g.add_node(("post", p["post_id"]), kind="post", sentiment=p["sentiment_label"])
    for _, c in comments.iterrows():
        g.add_node(("cmt", c["comment_id"]), kind="cmt", sentiment=c["sentiment_label"])
        g.add_edge(("post", c["post_id"]), ("cmt", c["comment_id"]))

    pos = nx.spring_layout(g, seed=42, k=0.35, iterations=80)

    post_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "post"]
    cmt_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "cmt"]
    post_colors = [COLORS[g.nodes[n]["sentiment"]] for n in post_nodes]
    cmt_colors = [COLORS[g.nodes[n]["sentiment"]] for n in cmt_nodes]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    nx.draw_networkx_edges(g, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=cmt_nodes, node_size=18,
                           node_color=cmt_colors, alpha=0.85, linewidths=0, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=post_nodes, node_size=110,
                           node_color=post_colors, edgecolors="black",
                           linewidths=0.8, ax=ax)
    ax.set_axis_off()
    ax.set_title("KIQ 3.3 — Post→comment network (force-directed)\n"
                 "Big rim = post, dot = comment, color = sentiment", fontsize=11)
    ax.legend(handles=LEGEND_PATCHES, loc="lower right", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- B. Post-column beeswarm -----------------------------------------

def beeswarm(posts: pd.DataFrame, comments: pd.DataFrame, out_path: Path,
             min_comments: int = 3) -> None:
    # Keep only posts with at least min_comments replies, so columns aren't empty.
    counts = comments.groupby("post_id").size()
    keep = counts[counts >= min_comments].index
    p = posts[posts["post_id"].isin(keep)].copy()
    c = comments[comments["post_id"].isin(keep)].copy()
    # Sort posts: positive (left) -> negative (right) by post sentiment_score.
    p = p.sort_values("sentiment_score", ascending=False).reset_index(drop=True)
    order = {pid: i for i, pid in enumerate(p["post_id"])}
    c["_x"] = c["post_id"].map(order)

    rng = np.random.default_rng(seed=42)
    c = c.copy()
    c["_xj"] = c["_x"] + rng.uniform(-0.32, 0.32, size=len(c))
    # y-jitter within the comment band [0.1, 1.0]
    c["_y"] = rng.uniform(0.1, 1.0, size=len(c))

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(p)), 5), dpi=120)

    # comment dots, colored by their own sentiment
    ax.scatter(c["_xj"], c["_y"],
               c=[COLORS[s] for s in c["sentiment_label"]],
               s=14, alpha=0.7, edgecolors="white", linewidths=0.3)

    # post markers below the swarm at y=-0.2, colored by the post's own sentiment
    ax.scatter(range(len(p)), [-0.2] * len(p),
               c=[COLORS[s] for s in p["sentiment_label"]],
               s=110, edgecolors="black", linewidths=0.6, marker="s")

    ax.axhline(0.0, color="black", lw=0.4)
    ax.set_xticks(range(len(p)))
    # show n on each column tick
    n_by_post = c.groupby("post_id").size()
    labels = [f"{n_by_post[pid]}" for pid in p["post_id"]]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Posts (sorted by post sentiment: ←  positive  …  negative  →); tick = n comments")
    ax.set_yticks([-0.2, 0.5], ["post", "comments"])
    ax.set_ylim(-0.45, 1.15)
    ax.set_xlim(-0.7, len(p) - 0.3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("KIQ 3.3 — Comments swarmed by parent post (post sentiment left→right)")
    ax.legend(handles=LEGEND_PATCHES, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- C. Post-vs-mean-comment scatter ---------------------------------

def scatter(posts: pd.DataFrame, comments: pd.DataFrame, out_path: Path) -> None:
    agg = (
        comments.groupby("post_id")
        .agg(n_comments=("sentiment_score", "size"),
             mean_cmt_score=("sentiment_score", "mean"))
        .reset_index()
    )
    df = posts.merge(agg, on="post_id", how="inner")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    # 45-degree reference (post sentiment == mean comment sentiment)
    ax.plot([-1, 1], [-1, 1], color="#aaa", lw=0.7, linestyle="--", zorder=1)
    ax.axhline(0, color="#ccc", lw=0.4); ax.axvline(0, color="#ccc", lw=0.4)

    sizes = 40 + 6 * df["n_comments"]
    colors = [COLORS[s] for s in df["sentiment_label"]]
    ax.scatter(df["sentiment_score"], df["mean_cmt_score"],
               s=sizes, c=colors, alpha=0.75, edgecolors="black", linewidths=0.6, zorder=3)

    # annotate the 5 biggest threads
    top = df.nlargest(5, "n_comments")
    for _, r in top.iterrows():
        ax.annotate(f"r/{r['subreddit']}\n(n={int(r['n_comments'])})",
                    (r["sentiment_score"], r["mean_cmt_score"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=8, color="#222")

    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Post sentiment score")
    ax.set_ylabel("Mean comment sentiment score (under that post)")
    ax.set_title("KIQ 3.3 — Each post: how its comments answered\n"
                 "Below the diagonal = comments more negative than the post")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=LEGEND_PATCHES, loc="upper left", frameon=False,
              title="Post sentiment", title_fontsize=9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    posts, comments = load()
    print(f"posts: {len(posts)} | comments: {len(comments)} (parent in set)")

    network(posts, comments, OUT_DIR / "kiq_3_3_thread_network.png")
    beeswarm(posts, comments, OUT_DIR / "kiq_3_3_thread_beeswarm.png")
    scatter(posts, comments, OUT_DIR / "kiq_3_3_thread_scatter.png")

    for name in ("kiq_3_3_thread_network.png", "kiq_3_3_thread_beeswarm.png", "kiq_3_3_thread_scatter.png"):
        print(f"Wrote: {OUT_DIR / name}")


if __name__ == "__main__":
    main()
