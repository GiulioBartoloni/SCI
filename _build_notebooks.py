"""Generate the three LLM-scored KIQ analysis notebooks under notebooks/.

Run: `uv run python _build_notebooks.py`. Re-run any time the cell content
needs to change; this overwrites the .ipynb files.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)

PREAMBLE = """\
import sys
from pathlib import Path

# Allow `import analysis_utils` and `import llm_utils` from notebooks/.
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from analysis_utils import (
    BRAND_KEYWORDS,
    detect_brands,
    explode_brands,
    has_any_brand,
    sentiment_distribution,
    weekly_volume_and_sentiment,
    monthly_volume_and_sentiment,
)
from llm_utils import (
    score_dataframe_with_cache,
    smoke_test,
    DEFAULT_MODEL,
    KIQ_TOPICS,
)

POSTS_PATH = ROOT / "data" / "processed" / "reddit_posts_clean.csv"
COMMENTS_PATH = ROOT / "data" / "processed" / "reddit_comments_clean.csv"
CACHE_DIR = ROOT / "data" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
"""

PREFLIGHT_CELL = """\
# Pre-flight: confirm Ollama + model are reachable BEFORE we start a long run.
import time
t0 = time.time()
_check = smoke_test()
print(f"Ollama OK ({DEFAULT_MODEL}) — single-call latency: {time.time()-t0:.1f}s")
print(f"  sample result: {_check}")
"""


def md(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell("\n".join(lines))


def write(nb: nbf.NotebookNode, path: Path) -> None:
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, path)
    print(f"  wrote {path}")


# --- shared scoring + relevance-filter cell template -----------------------
def llm_score_cells(kiq: str, *, kind_post_text: str, kind_comment_text: str) -> list:
    """Cells that run the LLM scoring + relevance filter for a KIQ.

    `kind_post_text` is the column on `posts_k` to score (e.g. 'text_combined' or
    a concatenated raw 'title + text'); same for comments.
    """
    return [
        md(
            "## LLM scoring (relevance + sentiment)",
            "",
            "Each post and comment is sent to Ollama (`gemma4-q6:latest`, `think=false`).",
            "The model returns `is_relevant`, `sentiment_label`, `sentiment_score`, and a brief `reason`.",
            "Results are cached to `data/llm_cache/` by item id, so re-running this cell is instant after the first pass.",
        ),
        code(PREFLIGHT_CELL),
        code(
            f"posts_scored = score_dataframe_with_cache(",
            f"    posts_k,",
            f"    text_col='{kind_post_text}',",
            f"    id_col='post_id',",
            f"    cache_path=CACHE_DIR / '{kiq.lower()}_posts.csv',",
            f"    kiq='{kiq}',",
            f"    kind='post',",
            f")",
            f"comments_scored = score_dataframe_with_cache(",
            f"    comments_k,",
            f"    text_col='{kind_comment_text}',",
            f"    id_col='comment_id',",
            f"    cache_path=CACHE_DIR / '{kiq.lower()}_comments.csv',",
            f"    kiq='{kiq}',",
            f"    kind='comment',",
            f")",
            f"posts_scored[['post_id', 'subreddit', 'is_relevant', 'sentiment_label', 'sentiment_score']].head()",
        ),
        md(
            "## Apply LLM relevance filter",
            "",
            "Drop items the model judged off-topic. The dropped sets are written to",
            f"`outputs/{kiq.lower()}/{kiq.lower()}_dropped_*.csv` for spot-checking.",
        ),
        code(
            "posts_before = len(posts_scored); comments_before = len(comments_scored)",
            "posts_dropped = posts_scored[posts_scored['is_relevant'] == False].copy()",
            "comments_dropped = comments_scored[comments_scored['is_relevant'] == False].copy()",
            "posts_scored = posts_scored[posts_scored['is_relevant'] == True].copy()",
            "comments_scored = comments_scored[comments_scored['is_relevant'] == True].copy()",
            "",
            "filter_log = pd.DataFrame([",
            "    {'slice': 'posts',    'n_before': posts_before,    'n_after': len(posts_scored)},",
            "    {'slice': 'comments', 'n_before': comments_before, 'n_after': len(comments_scored)},",
            "])",
            "print(f'Posts:    {posts_before:>5} -> {len(posts_scored):>5} after LLM relevance filter')",
            "print(f'Comments: {comments_before:>5} -> {len(comments_scored):>5} after LLM relevance filter')",
            "filter_log",
        ),
    ]


# ---------------------------------------------------------------------------
# KIQ 3.2 — Name Tag sentiment
# ---------------------------------------------------------------------------
def build_kiq_3_2() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        md(
            "# KIQ 3.2 — Sentiment on Meta's Name Tag facial recognition feature",
            "",
            "**Question:** What is the public sentiment on Meta's Name Tag (facial-recognition) feature on Reddit?",
            "",
            "**Pipeline:** for every KIQ_3_2 post and comment, ask the local LLM (`gemma4-q6` via Ollama)",
            "to (a) judge whether the item is *substantively* about Name Tag / facial recognition and",
            "(b) score sentiment toward that topic. Items the model marks as not-relevant are dropped.",
            "",
            "**Caveats**",
            "- Sample is small after relevance filtering. Cite direction + supporting quotes, not precise %.",
            "- LLM judgements are deterministic at temperature 0 but are not gold labels.",
            "  Spot-check `outputs/kiq_3_2/kiq_3_2_dropped_*.csv` to verify the filter is sensible.",
        ),
        md("## Setup"),
        code(PREAMBLE),
        code(
            "OUT_DIR = ROOT / 'outputs' / 'kiq_3_2'",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
            "KIQ = 'KIQ_3_2'",
        ),
        md("## Load and filter by KIQ label"),
        code(
            "posts = pd.read_csv(POSTS_PATH)",
            "comments = pd.read_csv(COMMENTS_PATH)",
            "",
            "# Build the text we send to the LLM. Use the original (pre-cleanup) title + body where available;",
            "# the cleaned text strips punctuation and case which hurts LLM comprehension.",
            "posts['_llm_text'] = (",
            "    posts['title'].fillna('').astype(str) + ' ' + posts['text'].fillna('').astype(str)",
            ").str.strip()",
            "comments['_llm_text'] = comments['body'].fillna('').astype(str)",
            "",
            "posts_k = posts[(posts['kiq_label'] == KIQ) & (posts['_llm_text'].str.len() > 0)].copy()",
            "comments_k = comments[(comments['kiq_label'] == KIQ) & (comments['_llm_text'].str.len() > 0)].copy()",
            "",
            "print(f'Posts (KIQ_3_2):    {len(posts_k):>5}')",
            "print(f'Comments (KIQ_3_2): {len(comments_k):>5}')",
            "print(f'Estimated first-run LLM time at 4s/call: ~{(len(posts_k)+len(comments_k))*4/60:.0f} minutes')",
        ),
        *llm_score_cells("KIQ_3_2",
                         kind_post_text="_llm_text", kind_comment_text="_llm_text"),
        md("## Sentiment label distribution"),
        code(
            "dist_posts = sentiment_distribution(posts_scored)",
            "dist_comments = sentiment_distribution(comments_scored)",
            "dist_combined = sentiment_distribution(",
            "    pd.concat([posts_scored[['sentiment_label']], comments_scored[['sentiment_label']]],",
            "              ignore_index=True)",
            ")",
            "dist_df = pd.DataFrame({'posts': dist_posts, 'comments': dist_comments, 'combined': dist_combined})",
            "dist_pct = (dist_df / dist_df.sum()) * 100",
            "dist_pct.round(1)",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)",
            "colors = {'positive': '#2a9d8f', 'neutral': '#adb5bd', 'negative': '#e76f51'}",
            "x = list(range(len(dist_pct.columns)))",
            "bottom = [0] * len(dist_pct.columns)",
            "for label in ['positive', 'neutral', 'negative']:",
            "    vals = dist_pct.loc[label].values",
            "    ax.bar(x, vals, bottom=bottom, label=label, color=colors[label])",
            "    bottom = [b + v for b, v in zip(bottom, vals)]",
            "ax.set_xticks(x); ax.set_xticklabels(dist_pct.columns)",
            "ax.set_ylim(0, 100); ax.set_ylabel('% of items')",
            "ax.set_title(f'KIQ 3.2 — Sentiment distribution on Name Tag content (LLM-scored)')",
            "ax.legend(loc='upper right', frameon=False)",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_3_2_distribution.png', dpi=300)",
            "plt.show()",
        ),
        md("## Mean sentiment score by subreddit"),
        code(
            "combined = pd.concat([",
            "    posts_scored[['subreddit', 'sentiment_score']].assign(_kind='post'),",
            "    comments_scored[['subreddit', 'sentiment_score']].assign(_kind='comment'),",
            "], ignore_index=True)",
            "by_sub = combined.groupby('subreddit').agg(",
            "    n=('sentiment_score', 'size'),",
            "    mean_score=('sentiment_score', 'mean'),",
            ")",
            "by_sub = by_sub[by_sub['n'] >= 3].sort_values('mean_score')",
            "by_sub",
        ),
        code(
            "if not by_sub.empty:",
            "    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(by_sub))), dpi=100)",
            "    colors_bar = ['#e76f51' if v < 0 else '#2a9d8f' for v in by_sub['mean_score']]",
            "    ax.barh(by_sub.index, by_sub['mean_score'], color=colors_bar)",
            "    ax.axvline(0, color='black', lw=0.6)",
            "    ax.set_xlim(-1, 1)",
            "    ax.set_xlabel('Mean LLM sentiment score (posts + comments)')",
            "    ax.set_title('KIQ 3.2 — Sentiment by subreddit (n >= 3)')",
            "    fig.tight_layout()",
            "    fig.savefig(OUT_DIR / 'kiq_3_2_by_subreddit.png', dpi=300)",
            "    plt.show()",
            "else:",
            "    print('No subreddit reached the n>=3 threshold; skipping chart.')",
        ),
        md("## Monthly trend"),
        code(
            "monthly_posts = monthly_volume_and_sentiment(posts_scored)",
            "monthly_comments = monthly_volume_and_sentiment(comments_scored)",
            "monthly_comments.head()",
        ),
        code(
            "fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=100)",
            "if not monthly_comments.empty:",
            "    ax1.bar(monthly_comments['month'], monthly_comments['n'],",
            "            color='#cfe2f3', label='comments / month')",
            "ax1.set_ylabel('Comment volume')",
            "ax1.tick_params(axis='x', rotation=45)",
            "ax2 = ax1.twinx()",
            "if not monthly_comments.empty:",
            "    ax2.plot(monthly_comments['month'], monthly_comments['mean_score'],",
            "             color='#1f3b73', marker='o', label='mean sentiment (comments)')",
            "ax2.axhline(0, color='black', lw=0.4, linestyle='--')",
            "ax2.set_ylim(-1, 1)",
            "ax2.set_ylabel('Mean LLM sentiment score')",
            "ax1.set_title('KIQ 3.2 — Monthly comment volume vs mean sentiment')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_3_2_monthly.png', dpi=300)",
            "plt.show()",
        ),
        md("## Top representative quotes"),
        code(
            "def top_quotes(df, text_col, kind, n=5):",
            "    if df.empty:",
            "        return pd.DataFrame()",
            "    cols = ['sentiment_score', 'sentiment_label', 'reason', 'subreddit', 'created_utc', text_col]",
            "    pos = df.nlargest(n, 'sentiment_score')[cols].assign(kind=kind, polarity='positive')",
            "    neg = df.nsmallest(n, 'sentiment_score')[cols].assign(kind=kind, polarity='negative')",
            "    out = pd.concat([pos, neg], ignore_index=True)",
            "    return out.rename(columns={text_col: 'text'})",
            "",
            "top = pd.concat([",
            "    top_quotes(posts_scored, '_llm_text', 'post'),",
            "    top_quotes(comments_scored, '_llm_text', 'comment'),",
            "], ignore_index=True)",
            "top['text'] = top['text'].astype(str).str.slice(0, 280)",
            "top",
        ),
        md("## Write outputs"),
        code(
            "summary = pd.DataFrame([",
            "    {'slice': 'posts', 'n': len(posts_scored),",
            "     'pct_positive': (posts_scored['sentiment_label'] == 'positive').mean() * 100,",
            "     'pct_neutral':  (posts_scored['sentiment_label'] == 'neutral').mean() * 100,",
            "     'pct_negative': (posts_scored['sentiment_label'] == 'negative').mean() * 100,",
            "     'mean_score':   posts_scored['sentiment_score'].mean()},",
            "    {'slice': 'comments', 'n': len(comments_scored),",
            "     'pct_positive': (comments_scored['sentiment_label'] == 'positive').mean() * 100,",
            "     'pct_neutral':  (comments_scored['sentiment_label'] == 'neutral').mean() * 100,",
            "     'pct_negative': (comments_scored['sentiment_label'] == 'negative').mean() * 100,",
            "     'mean_score':   comments_scored['sentiment_score'].mean()},",
            "])",
            "",
            "posts_scored.to_csv(OUT_DIR / 'kiq_3_2_scored_posts.csv', index=False)",
            "comments_scored.to_csv(OUT_DIR / 'kiq_3_2_scored_comments.csv', index=False)",
            "summary.to_csv(OUT_DIR / 'kiq_3_2_summary.csv', index=False)",
            "top.to_csv(OUT_DIR / 'kiq_3_2_top_quotes.csv', index=False)",
            "filter_log.to_csv(OUT_DIR / 'kiq_3_2_filter_log.csv', index=False)",
            "posts_dropped.to_csv(OUT_DIR / 'kiq_3_2_dropped_posts.csv', index=False)",
            "comments_dropped.to_csv(OUT_DIR / 'kiq_3_2_dropped_comments.csv', index=False)",
            "if not by_sub.empty:",
            "    by_sub.reset_index().to_csv(OUT_DIR / 'kiq_3_2_by_subreddit.csv', index=False)",
            "",
            "print('Wrote outputs to', OUT_DIR)",
            "for f in sorted(OUT_DIR.iterdir()):",
            "    print(' -', f.name)",
            "summary.round(2)",
        ),
    ]
    nb["cells"] = cells
    write(nb, NB_DIR / "kiq_3_2_name_tag_sentiment.ipynb")


# ---------------------------------------------------------------------------
# KIQ 3.3 — Privacy volume + tone over time
# ---------------------------------------------------------------------------
def build_kiq_3_3() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        md(
            "# KIQ 3.3 — Volume and tone of privacy-related Reddit content over time",
            "",
            "**Question:** How is privacy-related discussion of smart glasses framed (volume + tone) on Reddit, over time?",
            "",
            "**Pipeline:** LLM (`gemma4-q6` via Ollama) judges whether each KIQ_3_3 item is",
            "substantively about smart-glasses privacy and scores its sentiment.",
            "",
            "**Caveats**",
            "- Sample is broader than KIQ_3_2 (~2,300 items pre-filter), so percentages are more stable.",
            "- Weekly buckets are still noisy on a small corpus; smooth visually, not statistically.",
        ),
        md("## Setup"),
        code(PREAMBLE),
        code(
            "OUT_DIR = ROOT / 'outputs' / 'kiq_3_3'",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
            "KIQ = 'KIQ_3_3'",
        ),
        md("## Load and filter"),
        code(
            "posts = pd.read_csv(POSTS_PATH)",
            "comments = pd.read_csv(COMMENTS_PATH)",
            "posts['_llm_text'] = (",
            "    posts['title'].fillna('').astype(str) + ' ' + posts['text'].fillna('').astype(str)",
            ").str.strip()",
            "comments['_llm_text'] = comments['body'].fillna('').astype(str)",
            "",
            "posts_k = posts[(posts['kiq_label'] == KIQ) & (posts['_llm_text'].str.len() > 0)].copy()",
            "comments_k = comments[(comments['kiq_label'] == KIQ) & (comments['_llm_text'].str.len() > 0)].copy()",
            "",
            "print(f'Posts (KIQ_3_3):    {len(posts_k):>5}')",
            "print(f'Comments (KIQ_3_3): {len(comments_k):>5}')",
            "print(f'Estimated first-run LLM time at 4s/call: ~{(len(posts_k)+len(comments_k))*4/60:.0f} minutes')",
        ),
        *llm_score_cells("KIQ_3_3",
                         kind_post_text="_llm_text", kind_comment_text="_llm_text"),
        md("## Weekly volume and tone"),
        code(
            "weekly_posts = weekly_volume_and_sentiment(posts_scored).rename(",
            "    columns={'n': 'n_posts', 'mean_score': 'mean_score_posts'}",
            ")[['week', 'n_posts', 'mean_score_posts']]",
            "weekly_comments = weekly_volume_and_sentiment(comments_scored).rename(",
            "    columns={'n': 'n_comments', 'mean_score': 'mean_score_comments'}",
            ")[['week', 'n_comments', 'mean_score_comments']]",
            "weekly = pd.merge(weekly_posts, weekly_comments, on='week', how='outer').sort_values('week')",
            "weekly = weekly.fillna({'n_posts': 0, 'n_comments': 0})",
            "weekly.head()",
        ),
        code(
            "fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=100)",
            "ax1.bar(weekly['week'], weekly['n_comments'], width=5, color='#cfe2f3', label='comments')",
            "ax1.bar(weekly['week'], weekly['n_posts'], width=5, color='#5b9bd5', label='posts')",
            "ax1.set_ylabel('Volume per week')",
            "ax1.tick_params(axis='x', rotation=45)",
            "ax1.legend(loc='upper left', frameon=False)",
            "ax2 = ax1.twinx()",
            "ax2.plot(weekly['week'], weekly['mean_score_comments'], color='#1f3b73',",
            "         marker='o', markersize=3, lw=1.2, label='mean sentiment (comments)')",
            "ax2.axhline(0, color='black', lw=0.4, linestyle='--')",
            "ax2.set_ylim(-1, 1)",
            "ax2.set_ylabel('Mean LLM sentiment score')",
            "ax2.legend(loc='upper right', frameon=False)",
            "ax1.set_title('KIQ 3.3 — Weekly privacy-content volume and tone (LLM-scored)')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_3_3_volume_tone.png', dpi=300)",
            "plt.show()",
        ),
        md("## Subreddit breakdown"),
        code(
            "combined = pd.concat([",
            "    posts_scored.assign(_kind='post')[['subreddit', 'sentiment_score', '_kind']],",
            "    comments_scored.assign(_kind='comment')[['subreddit', 'sentiment_score', '_kind']],",
            "], ignore_index=True)",
            "sub_counts = combined['subreddit'].value_counts().head(10)",
            "sub_counts",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(sub_counts))), dpi=100)",
            "ax.barh(sub_counts.index[::-1], sub_counts.values[::-1], color='#5b9bd5')",
            "ax.set_xlabel('Items (posts + comments) in KIQ 3.3')",
            "ax.set_title('KIQ 3.3 — Where privacy framing concentrates (top 10 subs)')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_3_3_by_subreddit.png', dpi=300)",
            "plt.show()",
        ),
        md("## Sentiment label distribution"),
        code(
            "dist_posts = sentiment_distribution(posts_scored)",
            "dist_comments = sentiment_distribution(comments_scored)",
            "dist_df = pd.DataFrame({'posts': dist_posts, 'comments': dist_comments})",
            "dist_pct = (dist_df / dist_df.sum()) * 100",
            "dist_pct.round(1)",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4), dpi=100)",
            "colors = {'positive': '#2a9d8f', 'neutral': '#adb5bd', 'negative': '#e76f51'}",
            "x = list(range(len(dist_pct.columns)))",
            "bottom = [0] * len(dist_pct.columns)",
            "for label in ['positive', 'neutral', 'negative']:",
            "    vals = dist_pct.loc[label].values",
            "    ax.bar(x, vals, bottom=bottom, label=label, color=colors[label])",
            "    bottom = [b + v for b, v in zip(bottom, vals)]",
            "ax.set_xticks(x); ax.set_xticklabels(dist_pct.columns)",
            "ax.set_ylim(0, 100); ax.set_ylabel('% of items')",
            "ax.set_title('KIQ 3.3 — Sentiment label distribution')",
            "ax.legend(loc='upper right', frameon=False)",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_3_3_distribution.png', dpi=300)",
            "plt.show()",
        ),
        md("## Write outputs"),
        code(
            "posts_scored.to_csv(OUT_DIR / 'kiq_3_3_scored_posts.csv', index=False)",
            "comments_scored.to_csv(OUT_DIR / 'kiq_3_3_scored_comments.csv', index=False)",
            "weekly.to_csv(OUT_DIR / 'kiq_3_3_timeseries.csv', index=False)",
            "sub_counts.rename('n').to_frame().to_csv(OUT_DIR / 'kiq_3_3_by_subreddit.csv')",
            "dist_pct.to_csv(OUT_DIR / 'kiq_3_3_distribution.csv')",
            "filter_log.to_csv(OUT_DIR / 'kiq_3_3_filter_log.csv', index=False)",
            "posts_dropped.to_csv(OUT_DIR / 'kiq_3_3_dropped_posts.csv', index=False)",
            "comments_dropped.to_csv(OUT_DIR / 'kiq_3_3_dropped_comments.csv', index=False)",
            "",
            "print('Wrote outputs to', OUT_DIR)",
            "for f in sorted(OUT_DIR.iterdir()):",
            "    print(' -', f.name)",
        ),
    ]
    nb["cells"] = cells
    write(nb, NB_DIR / "kiq_3_3_privacy_volume_tone.ipynb")


# ---------------------------------------------------------------------------
# KIQ 2.2 — Brand share of voice
# ---------------------------------------------------------------------------
def build_kiq_2_2() -> None:
    nb = nbf.v4.new_notebook()
    cells = [
        md(
            "# KIQ 2.2 — Brand mention share across subreddits",
            "",
            "**Question:** What is each smart-glasses brand's share of mentions on Reddit (proxy for share of voice)?",
            "",
            "**Pipeline:**",
            "1. Filter to `kiq_label == 'KIQ_2_2'`.",
            "2. Pre-filter comments to those that contain at least one brand keyword (a no-brand comment cannot",
            "   contribute to share of voice).",
            "3. LLM (`gemma4-q6` via Ollama) judges substantive brand-relevance and scores sentiment.",
            "4. Keyword-based brand detection identifies which brand(s) each item mentions; the LLM result is",
            "   an *overall* sentiment for the item, not per-brand.",
            "",
            "**Caveat:** the per-brand sentiment column is the average overall-sentiment of items that mention",
            "the brand. It does NOT isolate sentiment toward a single brand within a multi-brand comparison.",
        ),
        md("## Setup"),
        code(PREAMBLE),
        code(
            "OUT_DIR = ROOT / 'outputs' / 'kiq_2_2'",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
            "KIQ = 'KIQ_2_2'",
        ),
        md("## Load + filter + brand pre-filter on comments"),
        code(
            "posts = pd.read_csv(POSTS_PATH)",
            "comments = pd.read_csv(COMMENTS_PATH)",
            "posts['_llm_text'] = (",
            "    posts['title'].fillna('').astype(str) + ' ' + posts['text'].fillna('').astype(str)",
            ").str.strip()",
            "comments['_llm_text'] = comments['body'].fillna('').astype(str)",
            "",
            "posts_k = posts[(posts['kiq_label'] == KIQ) & (posts['_llm_text'].str.len() > 0)].copy()",
            "comments_k = comments[(comments['kiq_label'] == KIQ) & (comments['_llm_text'].str.len() > 0)].copy()",
            "",
            "# Brand-keyword pre-filter on comments only (posts already retrieved by brand-themed queries).",
            "comments_k = comments_k[comments_k['_llm_text'].apply(has_any_brand)].copy()",
            "",
            "print(f'Posts (KIQ_2_2):              {len(posts_k):>6}')",
            "print(f'Comments (KIQ_2_2, brand-pre-filtered): {len(comments_k):>6}')",
            "print(f'Estimated first-run LLM time at 4s/call: ~{(len(posts_k)+len(comments_k))*4/60:.0f} minutes')",
        ),
        *llm_score_cells("KIQ_2_2",
                         kind_post_text="_llm_text", kind_comment_text="_llm_text"),
        md("## Re-detect brands on the filtered set"),
        code(
            "# Posts already have a 'brand' column from label_kiq.py, but recompute on the LLM-relevant subset",
            "# so the brand text matches what the LLM saw.",
            "posts_scored['brand'] = posts_scored['_llm_text'].map(detect_brands)",
            "comments_scored['brand'] = comments_scored['_llm_text'].map(detect_brands)",
            "comments_scored['brand'].value_counts().head(10)",
        ),
        md("## Brand share of voice"),
        code(
            "brand_posts = explode_brands(posts_scored, brand_col='brand')",
            "brand_comments = explode_brands(comments_scored, brand_col='brand')",
            "",
            "share = pd.DataFrame({",
            "    'n_posts': brand_posts['brand'].value_counts(),",
            "    'n_comments': brand_comments['brand'].value_counts(),",
            "}).fillna(0).astype(int)",
            "share['share_pct_posts']    = share['n_posts']    / share['n_posts'].sum()    * 100",
            "share['share_pct_comments'] = share['n_comments'] / share['n_comments'].sum() * 100",
            "share = share.sort_values('n_comments', ascending=False)",
            "share.round(2)",
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100, sharey=True)",
            "for ax, col, title in [",
            "    (axes[0], 'share_pct_posts', 'Posts'),",
            "    (axes[1], 'share_pct_comments', 'Comments'),",
            "]:",
            "    s = share[col].sort_values()",
            "    ax.barh(s.index, s.values, color='#5b9bd5')",
            "    ax.set_title(f'{title} — share of brand mentions (%)')",
            "    ax.set_xlabel('% of brand mentions')",
            "fig.suptitle('KIQ 2.2 — Brand share of voice (LLM-relevance filtered)')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_2_2_share.png', dpi=300)",
            "plt.show()",
        ),
        md("## Brand × subreddit"),
        code(
            "brand_sub = (brand_comments.groupby(['subreddit', 'brand']).size()",
            "             .rename('n').reset_index())",
            "top_subs = (brand_sub.groupby('subreddit')['n'].sum()",
            "            .sort_values(ascending=False).head(8).index.tolist())",
            "matrix = (brand_sub[brand_sub['subreddit'].isin(top_subs)]",
            "          .pivot(index='subreddit', columns='brand', values='n')",
            "          .reindex(top_subs).fillna(0))",
            "matrix",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(10, 5), dpi=100)",
            "matrix.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', width=0.8)",
            "ax.set_ylabel('Brand mentions in comments')",
            "ax.set_xlabel('Subreddit')",
            "ax.set_title('KIQ 2.2 — Brand mentions by subreddit (top 8 subs, comments)')",
            "ax.legend(title='brand', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)",
            "plt.xticks(rotation=30, ha='right')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_2_2_by_subreddit.png', dpi=300)",
            "plt.show()",
        ),
        md("## Sentiment per brand (overall-sentiment of items that mention the brand)"),
        code(
            "brand_sent = pd.concat([",
            "    brand_posts[['brand', 'sentiment_score', 'sentiment_label']].assign(_kind='post'),",
            "    brand_comments[['brand', 'sentiment_score', 'sentiment_label']].assign(_kind='comment'),",
            "], ignore_index=True)",
            "per_brand = brand_sent.groupby('brand').agg(",
            "    n=('sentiment_score', 'size'),",
            "    mean_score=('sentiment_score', 'mean'),",
            "    pct_positive=('sentiment_label', lambda s: (s == 'positive').mean() * 100),",
            "    pct_negative=('sentiment_label', lambda s: (s == 'negative').mean() * 100),",
            ").sort_values('mean_score', ascending=False)",
            "per_brand.round(3)",
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)",
            "colors_bar = ['#e76f51' if v < 0 else '#2a9d8f' for v in per_brand['mean_score']]",
            "ax.barh(per_brand.index[::-1], per_brand['mean_score'].values[::-1], color=colors_bar[::-1])",
            "ax.axvline(0, color='black', lw=0.6)",
            "ax.set_xlim(-1, 1)",
            "ax.set_xlabel('Mean LLM sentiment score (posts + comments)')",
            "ax.set_title('KIQ 2.2 — Mean overall sentiment per brand')",
            "fig.tight_layout()",
            "fig.savefig(OUT_DIR / 'kiq_2_2_sentiment_per_brand.png', dpi=300)",
            "plt.show()",
        ),
        md("## Write outputs"),
        code(
            "posts_scored.to_csv(OUT_DIR / 'kiq_2_2_scored_posts.csv', index=False)",
            "comments_scored.to_csv(OUT_DIR / 'kiq_2_2_scored_comments.csv', index=False)",
            "share.to_csv(OUT_DIR / 'kiq_2_2_brand_share.csv')",
            "brand_sub.to_csv(OUT_DIR / 'kiq_2_2_brand_subreddit.csv', index=False)",
            "per_brand.to_csv(OUT_DIR / 'kiq_2_2_sentiment_per_brand.csv')",
            "filter_log.to_csv(OUT_DIR / 'kiq_2_2_filter_log.csv', index=False)",
            "posts_dropped.to_csv(OUT_DIR / 'kiq_2_2_dropped_posts.csv', index=False)",
            "comments_dropped.to_csv(OUT_DIR / 'kiq_2_2_dropped_comments.csv', index=False)",
            "",
            "print('Wrote outputs to', OUT_DIR)",
            "for f in sorted(OUT_DIR.iterdir()):",
            "    print(' -', f.name)",
        ),
    ]
    nb["cells"] = cells
    write(nb, NB_DIR / "kiq_2_2_brand_share.ipynb")


def main() -> None:
    print("Building notebooks:")
    build_kiq_3_2()
    build_kiq_3_3()
    build_kiq_2_2()


if __name__ == "__main__":
    main()
