"""Clean post and comment text for downstream NLP."""
import re
import sys
from pathlib import Path

import pandas as pd

POSTS_IN = Path("data/raw/reddit_posts.csv")
COMMENTS_IN = Path("data/raw/reddit_comments.csv")
POSTS_OUT = Path("data/processed/reddit_posts_clean.csv")
COMMENTS_OUT = Path("data/processed/reddit_comments_clean.csv")

DROP_TOKENS = {"[deleted]", "[removed]", "[ deleted ]", "[ removed ]", "null", ""}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MD_FORMATTING_RE = re.compile(r"[*_`>#~]+")
WHITESPACE_RE = re.compile(r"\s+")
SPECIALS_RE = re.compile(r"[^a-z0-9\s.,!?'-]")


def clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    s = value
    s = MD_LINK_RE.sub(r"\1", s)
    s = URL_RE.sub(" ", s)
    s = s.lower()
    s = MD_FORMATTING_RE.sub(" ", s)
    s = SPECIALS_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def is_dead(value) -> bool:
    if not isinstance(value, str):
        return True
    return value.strip().lower() in DROP_TOKENS


def clean_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["title_clean"] = df["title"].map(clean_text)
    df["text_clean"] = df["text"].apply(lambda v: "" if is_dead(v) else clean_text(v))
    df["text_combined"] = (df["title_clean"] + " " + df["text_clean"]).str.strip()
    df = df[df["text_combined"].str.len() > 0]
    return df


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = df["body"].fillna("").astype(str)
    df = df[~df["body"].map(is_dead)]
    df["body_clean"] = df["body"].map(clean_text)
    df = df[df["body_clean"].str.len() > 0]
    return df


def main() -> None:
    POSTS_OUT.parent.mkdir(parents=True, exist_ok=True)

    if POSTS_IN.exists():
        posts = pd.read_csv(POSTS_IN)
        print(f"Posts in: {len(posts)}")
        posts_clean = clean_posts(posts)
        posts_clean.to_csv(POSTS_OUT, index=False)
        print(f"Posts out: {len(posts_clean)} -> {POSTS_OUT}")
    else:
        print(f"[skip] {POSTS_IN} not found", file=sys.stderr)

    if COMMENTS_IN.exists():
        comments = pd.read_csv(COMMENTS_IN)
        print(f"Comments in: {len(comments)}")
        comments_clean = clean_comments(comments)
        comments_clean.to_csv(COMMENTS_OUT, index=False)
        print(f"Comments out: {len(comments_clean)} -> {COMMENTS_OUT}")
    else:
        print(f"[skip] {COMMENTS_IN} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
