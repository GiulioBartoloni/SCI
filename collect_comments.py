"""Collect Reddit comments for each post in data/raw/reddit_posts.csv via Arctic Shift."""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_BASE = "https://arctic-shift.photon-reddit.com/api"
COMMENTS_ENDPOINT = f"{API_BASE}/comments/search"

POSTS_PATH = Path("data/raw/reddit_posts.csv")
OUTPUT_PATH = Path("data/raw/reddit_comments.csv")

PAGE_LIMIT = 100
COMMENTS_PER_POST = 500


def fetch_page(params: dict, retries: int = 3) -> list[dict] | None:
    for attempt in range(retries):
        try:
            r = requests.get(COMMENTS_ENDPOINT, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"[warn] network error ({exc}); retrying", file=sys.stderr)
            time.sleep(5)
            continue
        if r.status_code == 429:
            time.sleep(10)
            continue
        if r.status_code != 200:
            print(f"[error] HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return None
        body = r.json()
        if body.get("error"):
            print(f"[error] api: {body['error']} params={params}", file=sys.stderr)
            return None
        if "data" not in body:
            print(f"[error] unexpected schema: {body!r}", file=sys.stderr)
            sys.exit(1)
        return body.get("data") or []
    return None


def collect_for_post(post_id: str) -> list[dict]:
    collected: dict[str, dict] = {}
    after = 0
    while len(collected) < COMMENTS_PER_POST:
        params = {
            "link_id": f"t3_{post_id}",
            "sort": "asc",
            "limit": PAGE_LIMIT,
        }
        if after:
            params["after"] = after
        page = fetch_page(params)
        if page is None or not page:
            break
        for c in page:
            cid = c.get("id")
            if not cid or cid in collected:
                continue
            collected[cid] = c
            if len(collected) >= COMMENTS_PER_POST:
                break
        if len(page) < PAGE_LIMIT:
            break
        last_ts = page[-1].get("created_utc")
        if not isinstance(last_ts, (int, float)):
            break
        after = int(last_ts) + 1
        time.sleep(0.5)
    return list(collected.values())


def to_row(c: dict, post_id: str) -> dict:
    ts = c.get("created_utc")
    date_str = (
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if isinstance(ts, (int, float)) else ""
    )
    return {
        "comment_id": c.get("id", ""),
        "post_id": post_id,
        "body": c.get("body") or "",
        "score": c.get("score", 0),
        "created_utc": date_str,
        "subreddit": c.get("subreddit", ""),
    }


def main() -> None:
    if not POSTS_PATH.exists():
        sys.exit(f"missing {POSTS_PATH}; run collect_posts.py first")

    posts = pd.read_csv(POSTS_PATH)
    targets = posts[posts["num_comments"].fillna(0).astype(int) > 0]
    targets = targets.drop_duplicates(subset="post_id")
    print(f"Fetching comments for {len(targets)} posts (out of {len(posts)})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["comment_id", "post_id", "body", "score", "created_utc", "subreddit"]
    written = 0
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for post_id in tqdm(targets["post_id"].tolist(), desc="posts"):
            comments = collect_for_post(post_id)
            for c in comments:
                writer.writerow(to_row(c, post_id))
                written += 1
            time.sleep(0.5)

    print(f"\nWrote {written} comments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
