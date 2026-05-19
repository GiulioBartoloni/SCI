"""KIQ 3.3 — Co-occurrence (bigrams + unigrams) on negative privacy comments.

Reads the LLM-scored output from `outputs/kiq_3_3/` (no re-scoring required) and
writes top-bigram / top-unigram charts and CSVs to the same folder.

Two passes are produced so the storyteller can choose:
  *_full       — every token kept (meta / glasses dominate, but useful as ground truth)
  *_themed     — domain "frame" words filtered out so themes (consent, recording,
                 surveillance, faces, kids ...) surface visually
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_DIR = ROOT / "outputs" / "kiq_3_3"
OUT_DIR = IN_DIR  # write back alongside existing kiq_3_3 outputs

# Standard English stopwords, expanded with low-information Reddit fillers.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "ma", "me", "might", "more", "most", "must", "my", "myself", "need",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",
    "she", "should", "shouldn", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
    "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "won", "would", "wouldn", "y", "you",
    "your", "yours", "yourself", "yourselves",
    # Reddit / web fillers
    "like", "get", "got", "going", "go", "going", "still", "even", "also", "back",
    "us", "u", "im", "ill", "thats", "youre", "theyre", "dont", "doesnt", "didnt",
    "isnt", "wasnt", "wont", "cant", "id", "ive", "youll", "youd", "really",
    "actually", "maybe", "probably", "literally", "basically", "kind", "lot",
    "much", "many", "thing", "things", "way", "ways", "people", "person",
    "someone", "anyone", "everyone", "something", "anything", "everything",
    "nothing", "well", "yeah", "yes", "ok", "okay", "sure", "right", "good", "bad",
    "make", "made", "makes", "making", "want", "wants", "wanted", "use", "used",
    "using", "say", "said", "says", "saying", "think", "thought", "thinking",
    "know", "knows", "knew", "see", "seen", "saw", "look", "looks", "looking",
    "one", "two", "three", "first", "last", "new", "old", "long", "short",
    "ever", "never", "always", "sometimes", "often", "around", "another",
    "comment", "post", "reply", "edit", "deleted", "removed", "amp", "http",
    "https", "www", "com", "url", "ie", "eg", "etc", "vs", "via",
    "thing", "stuff", "guy", "guys", "anybody", "everybody", "nobody",
    "able", "let", "lets", "took", "take", "takes", "taking", "give", "gives",
    "given", "giving", "come", "came", "comes", "coming", "tell", "told", "tells",
    "feel", "felt", "feeling", "find", "found", "finds", "finding", "work",
    "works", "worked", "working", "try", "tried", "tries", "trying",
}

# Domain "frame" words removed in the *_themed* pass so the *content* themes show.
FRAME_WORDS = {
    "meta", "glasses", "glass", "smart", "ai", "ray", "ban", "rayban", "ray-ban",
    "metas", "meta's", "facebook", "zuckerberg", "zuck", "tech", "company",
    "device", "devices", "wear", "wearing", "wears", "phone", "phones",
}

TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")  # words only, keep contractions/hyphens

# Contractions and possessives — uninformative on their own, drop after tokenization.
CONTRACTIONS = {
    "it's", "i'm", "don't", "that's", "you're", "they're", "we're", "i've",
    "i'd", "i'll", "you've", "you'll", "you'd", "we've", "we'll", "we'd",
    "they've", "they'll", "they'd", "he's", "she's", "there's", "here's",
    "what's", "who's", "where's", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "can't", "haven't", "hasn't", "hadn't", "let's", "ain't",
}


def tokenize(text: str, *, drop_frame: bool = False) -> list[str]:
    if not isinstance(text, str):
        return []
    toks = TOKEN_RE.findall(text.lower())
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in STOPWORDS or t in CONTRACTIONS:
            continue
        if drop_frame and t in FRAME_WORDS:
            continue
        out.append(t)
    return out


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


def top_terms(corpus: list[str], *, drop_frame: bool, top_n: int = 25):
    uni_counter: Counter[str] = Counter()
    bi_counter: Counter[tuple[str, str]] = Counter()
    for text in corpus:
        toks = tokenize(text, drop_frame=drop_frame)
        uni_counter.update(toks)
        bi_counter.update(bigrams(toks))
    uni = pd.DataFrame(uni_counter.most_common(top_n), columns=["term", "count"])
    bi = pd.DataFrame(
        [(f"{a} {b}", c) for (a, b), c in bi_counter.most_common(top_n)],
        columns=["bigram", "count"],
    )
    return uni, bi


def hbar(df: pd.DataFrame, value_col: str, label_col: str, title: str, out_path: Path):
    df = df.iloc[::-1]  # plot top-at-top
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(df))), dpi=100)
    ax.barh(df[label_col], df[value_col], color="#e76f51")
    ax.set_xlabel("Occurrences")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    comments = pd.read_csv(IN_DIR / "kiq_3_3_scored_comments.csv")
    posts = pd.read_csv(IN_DIR / "kiq_3_3_scored_posts.csv")

    neg_comments = comments[comments["sentiment_label"] == "negative"]
    neg_posts = posts[posts["sentiment_label"] == "negative"]
    print(f"Negative comments: {len(neg_comments)}  |  Negative posts: {len(neg_posts)}")

    corpus_comments = neg_comments["body_clean"].dropna().tolist()
    # Posts: combine title + body cleaned text (body_clean exists for comments only; for
    # posts we already produced `text_combined`).
    corpus_posts = (
        neg_posts.get("text_combined", neg_posts.get("title_clean", pd.Series(dtype=str)))
        .dropna()
        .tolist()
    )
    corpus = corpus_comments + corpus_posts
    print(f"Total negative documents: {len(corpus)}")

    for tag, drop_frame in (("full", False), ("themed", True)):
        uni, bi = top_terms(corpus, drop_frame=drop_frame, top_n=25)
        uni.to_csv(OUT_DIR / f"kiq_3_3_unigrams_neg_{tag}.csv", index=False)
        bi.to_csv(OUT_DIR / f"kiq_3_3_bigrams_neg_{tag}.csv", index=False)
        hbar(
            uni.head(20),
            value_col="count",
            label_col="term",
            title=(
                "KIQ 3.3 — Top unigrams in negative privacy content"
                + (" (frame words removed)" if drop_frame else "")
            ),
            out_path=OUT_DIR / f"kiq_3_3_unigrams_neg_{tag}.png",
        )
        hbar(
            bi.head(20),
            value_col="count",
            label_col="bigram",
            title=(
                "KIQ 3.3 — Top bigrams in negative privacy content"
                + (" (frame words removed)" if drop_frame else "")
            ),
            out_path=OUT_DIR / f"kiq_3_3_bigrams_neg_{tag}.png",
        )
        print(f"[{tag}] top bigrams:")
        print(bi.head(10).to_string(index=False))
        print()

    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
