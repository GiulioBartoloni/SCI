"""KIQ 2.2 — Co-occurrence on Meta-brand items, split by polarity x kind.

Same tokenizer as kiq_3_3_bigrams.py, applied to the KIQ 2.2 corpus (brand mentions,
not pre-filtered to privacy). Four slices:

  *_pos_posts_*    — positive Meta-brand posts
  *_neg_posts_*    — negative Meta-brand posts
  *_pos_comments_* — positive Meta-brand comments
  *_neg_comments_* — negative Meta-brand comments

The positive slices reveal the praise lexicon (what people *like* about Meta's
glasses). The negative slices reveal the complaint lexicon (what they *don't*).
Splitting posts vs comments matters because posts skew toward news/headlines
while comments skew toward lived experience, so the lexicons differ.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kiq_3_3_bigrams import hbar, top_terms

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_2_2"
OUT_DIR = IN_DIR

# Loose definition of a "privacy" term — used only for the headline KPI line at the
# bottom of the script, not for filtering.
PRIVACY_TERMS = {
    "privacy", "surveillance", "consent", "recording", "record", "recorded",
    "camera", "cameras", "facial", "recognition", "tracking", "spy", "spying",
    "creep", "creepy", "data",
}


def filter_meta(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["brand"].fillna("").str.contains(r"\bmeta\b", regex=True)]


def corpus_for(df: pd.DataFrame, text_col_priority: list[str]) -> list[str]:
    for c in text_col_priority:
        if c in df.columns:
            return df[c].dropna().astype(str).tolist()
    raise KeyError(f"None of {text_col_priority} in df.columns")


def privacy_kpi(docs: list[str]) -> tuple[int, float]:
    """Return (count, share) of documents containing any PRIVACY_TERMS keyword."""
    hits = sum(
        1
        for d in docs
        if any(kw in d.lower() for kw in PRIVACY_TERMS)
    )
    share = hits / len(docs) * 100 if docs else 0.0
    return hits, share


def render(corpus: list[str], slice_tag: str, title_suffix: str) -> None:
    uni, bi = top_terms(corpus, drop_frame=True, top_n=25)
    uni.to_csv(OUT_DIR / f"kiq_2_2_unigrams_{slice_tag}_themed.csv", index=False)
    bi.to_csv(OUT_DIR / f"kiq_2_2_bigrams_{slice_tag}_themed.csv", index=False)
    hbar(
        uni.head(20),
        value_col="count",
        label_col="term",
        title=f"KIQ 2.2 — Top unigrams in {title_suffix} (frame words removed)",
        out_path=OUT_DIR / f"kiq_2_2_unigrams_{slice_tag}_themed.png",
    )
    hbar(
        bi.head(20),
        value_col="count",
        label_col="bigram",
        title=f"KIQ 2.2 — Top bigrams in {title_suffix} (frame words removed)",
        out_path=OUT_DIR / f"kiq_2_2_bigrams_{slice_tag}_themed.png",
    )
    print(f"[{slice_tag}] top bigrams:")
    print(bi.head(10).to_string(index=False))
    print()


POST_TEXT_COLS = ["text_combined", "title_clean", "_llm_text"]
COMMENT_TEXT_COLS = ["body_clean", "_llm_text"]


def main() -> None:
    posts = pd.read_csv(IN_DIR / "kiq_2_2_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_2_2_scored_comments.csv")
    posts = posts[posts["is_relevant"] == True]
    comments = comments[comments["is_relevant"] == True]

    meta_posts = filter_meta(posts)
    meta_comments = filter_meta(comments)
    print(f"Meta posts: {len(meta_posts)} | Meta comments: {len(meta_comments)}")
    print()

    slices = [
        ("pos_posts",    meta_posts[meta_posts["sentiment_label"] == "positive"],    POST_TEXT_COLS,    "positive Meta-brand posts"),
        ("neg_posts",    meta_posts[meta_posts["sentiment_label"] == "negative"],    POST_TEXT_COLS,    "negative Meta-brand posts"),
        ("pos_comments", meta_comments[meta_comments["sentiment_label"] == "positive"], COMMENT_TEXT_COLS, "positive Meta-brand comments"),
        ("neg_comments", meta_comments[meta_comments["sentiment_label"] == "negative"], COMMENT_TEXT_COLS, "negative Meta-brand comments"),
    ]

    kpi_rows = []
    for tag, df, text_cols, title in slices:
        print(f"=== {tag} (n={len(df)}) ===")
        if len(df) < 20:
            print(f"  skipped: sample too small ({len(df)} < 20)")
            continue
        corpus = corpus_for(df, text_cols)
        render(corpus, tag, title)
        hits, share = privacy_kpi(corpus)
        kpi_rows.append({
            "slice": tag,
            "n": len(corpus),
            "with_privacy_term": hits,
            "share_pct": round(share, 1),
        })

    kpi = pd.DataFrame(kpi_rows)
    kpi.to_csv(OUT_DIR / "kiq_2_2_privacy_share.csv", index=False)
    print("Privacy-term presence by slice:")
    print(kpi.to_string(index=False))


if __name__ == "__main__":
    main()
