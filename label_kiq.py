"""Tag cleaned posts with KIQ label and (for KIQ_2_2) brand mentions."""
import sys
from pathlib import Path

import pandas as pd

from analysis_utils import BRAND_KEYWORDS, detect_brands

POSTS_IN = Path("data/processed/reddit_posts_clean.csv")
COMMENTS_IN = Path("data/processed/reddit_comments_clean.csv")
POSTS_OUT = Path("data/processed/reddit_posts_clean.csv")
COMMENTS_OUT = Path("data/processed/reddit_comments_clean.csv")

QUERY_TO_KIQ = {
    "Meta Name Tag glasses": "KIQ_3_2",
    "Ray-Ban Meta facial recognition": "KIQ_3_2",
    "Meta smart glasses privacy": "KIQ_3_2",
    "Name Tag feature Meta": "KIQ_3_2",
    "Meta glasses surveillance": "KIQ_3_3",
    "Ray-Ban Meta camera privacy": "KIQ_3_3",
    "smart glasses privacy": "KIQ_3_3",
    "Meta glasses data collection": "KIQ_3_3",
    "Ray-Ban Meta glasses": "KIQ_2_2",
    "Apple smart glasses": "KIQ_2_2",
    "Google Warby Parker glasses": "KIQ_2_2",
    "Snap Spectacles 2026": "KIQ_2_2",
    "Xreal glasses": "KIQ_2_2",
    "Xiaomi smart glasses": "KIQ_2_2",
}


def label_posts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kiq_label"] = df["query"].map(QUERY_TO_KIQ).fillna("")
    text_col = "text_combined" if "text_combined" in df.columns else "title"
    df["brand"] = ""
    mask = df["kiq_label"] == "KIQ_2_2"
    df.loc[mask, "brand"] = df.loc[mask, text_col].map(detect_brands)
    return df


def label_comments(comments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """Propagate kiq_label from each comment's parent post."""
    if "post_id" not in comments.columns:
        return comments
    lookup = posts.drop_duplicates(subset="post_id").set_index("post_id")["kiq_label"]
    out = comments.copy()
    out["kiq_label"] = out["post_id"].map(lookup).fillna("")
    return out


def main() -> None:
    if not POSTS_IN.exists():
        sys.exit(f"missing {POSTS_IN}; run preprocess.py first")

    posts = pd.read_csv(POSTS_IN)
    posts_labeled = label_posts(posts)
    posts_labeled.to_csv(POSTS_OUT, index=False)
    print(f"Labeled {len(posts_labeled)} posts -> {POSTS_OUT}")
    print("\nKIQ counts (posts):")
    print(posts_labeled["kiq_label"].value_counts().to_string())
    brand_only = posts_labeled[posts_labeled["kiq_label"] == "KIQ_2_2"]
    if len(brand_only):
        non_empty = brand_only[brand_only["brand"].astype(str).str.len() > 0]
        print(f"\nKIQ_2_2 posts with at least one brand match: "
              f"{len(non_empty)}/{len(brand_only)}")

    if COMMENTS_IN.exists():
        comments = pd.read_csv(COMMENTS_IN)
        comments_labeled = label_comments(comments, posts_labeled)
        comments_labeled.to_csv(COMMENTS_OUT, index=False)
        print(f"\nLabeled {len(comments_labeled)} comments -> {COMMENTS_OUT}")
        print("\nKIQ counts (comments):")
        print(comments_labeled["kiq_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
