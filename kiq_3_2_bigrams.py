"""KIQ 3.2 — Co-occurrence on Name Tag-specific items.

Same tokenizer as kiq_3_3_bigrams.py, applied to the KIQ 3.2 corpus
(items the LLM tagged as substantively about Name Tag / facial recognition).

Caveat: this corpus is small (~146 relevant items). Unigrams are stable;
bigrams are noisier. Read the unigram outputs as the primary signal.

Two slices:
  *_all_nametag_*    every relevant Name Tag item, any sentiment
                     (the corpus is ~82% negative, so this is dominated by complaint)
  *_neg_nametag_*    negative-only items, for parity with KIQ 3.3 / KIQ 2.2 neg passes

Frame words include the topic itself (name, tag, facial, recognition, face) so
the *themed* pass surfaces what people say AROUND the feature, not the feature.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import kiq_3_3_bigrams as bg

# Extend the shared FRAME_WORDS set with Name Tag-specific topic words.
NAMETAG_FRAME_WORDS = {
    "name", "tag", "names", "tags", "facial", "recognition",
    "recognize", "recognise", "recognizes", "recognised", "recognized",
    "face", "faces", "feature", "features",
}
bg.FRAME_WORDS = bg.FRAME_WORDS | NAMETAG_FRAME_WORDS

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_2"
OUT_DIR = IN_DIR


def corpus_for_posts(df: pd.DataFrame) -> list[str]:
    # Posts in 3.2 keep title+body in _llm_text (no body_clean column).
    return df["_llm_text"].dropna().astype(str).tolist()


def corpus_for_comments(df: pd.DataFrame) -> list[str]:
    return df["body_clean"].dropna().astype(str).tolist()


def render(corpus: list[str], slice_tag: str, title_suffix: str) -> None:
    for view, drop_frame in (("full", False), ("themed", True)):
        uni, bi = bg.top_terms(corpus, drop_frame=drop_frame, top_n=25)
        uni.to_csv(OUT_DIR / f"kiq_3_2_unigrams_{slice_tag}_{view}.csv", index=False)
        bi.to_csv(OUT_DIR / f"kiq_3_2_bigrams_{slice_tag}_{view}.csv", index=False)
        suffix = " (frame words removed)" if drop_frame else ""
        bg.hbar(
            uni.head(20),
            value_col="count",
            label_col="term",
            title=f"KIQ 3.2 — Top unigrams in {title_suffix}{suffix}",
            out_path=OUT_DIR / f"kiq_3_2_unigrams_{slice_tag}_{view}.png",
        )
        bg.hbar(
            bi.head(20),
            value_col="count",
            label_col="bigram",
            title=f"KIQ 3.2 — Top bigrams in {title_suffix}{suffix}",
            out_path=OUT_DIR / f"kiq_3_2_bigrams_{slice_tag}_{view}.png",
        )
        print(f"[{slice_tag}/{view}] top bigrams:")
        print(bi.head(10).to_string(index=False))
        print()


def main() -> None:
    posts = pd.read_csv(IN_DIR / "kiq_3_2_scored_posts.csv")
    comments = pd.read_csv(IN_DIR / "kiq_3_2_scored_comments.csv")
    posts = posts[posts["is_relevant"] == True]
    comments = comments[comments["is_relevant"] == True]
    print(f"Posts: {len(posts)} | Comments: {len(comments)}")
    print()

    all_corpus = corpus_for_comments(comments) + corpus_for_posts(posts)
    neg_posts = posts[posts["sentiment_label"] == "negative"]
    neg_comments = comments[comments["sentiment_label"] == "negative"]
    neg_corpus = corpus_for_comments(neg_comments) + corpus_for_posts(neg_posts)

    print(f"All-Name Tag corpus: {len(all_corpus)} docs")
    print(f"Negative-Name Tag corpus: {len(neg_corpus)} docs")
    print()

    render(all_corpus, "all_nametag", "all Name Tag discussion")
    render(neg_corpus, "neg_nametag", "negative Name Tag discussion")


if __name__ == "__main__":
    main()
