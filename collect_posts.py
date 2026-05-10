"""Collect Reddit posts via Arctic Shift API.

Iterates each query across a curated subreddit list (Arctic Shift requires
`subreddit` or `author` to be set when keyword-searching), paginates by
ascending `created_utc`, and stops at POSTS_PER_QUERY or when a page returns
fewer than PAGE_LIMIT results.
"""
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://arctic-shift.photon-reddit.com/api"
POSTS_ENDPOINT = f"{API_BASE}/posts/search"

AFTER_TS = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
BEFORE_TS = int(datetime.now(tz=timezone.utc).timestamp())

SUBREDDITS = [
    "technology", "gadgets", "futurology", "privacy",
    "smartglasses", "augmentedreality", "RayBanMeta", "MetaQuest",
    "oculus", "hardware", "technews", "apple", "Android",
    "snapchat", "Spectacles", "Xreal",
]

QUERIES_BY_KIQ = {
    "KIQ_3_2": [
        "Meta Name Tag glasses",
        "Ray-Ban Meta facial recognition",
        "Meta smart glasses privacy",
        "Name Tag feature Meta",
    ],
    "KIQ_3_3": [
        "Meta glasses surveillance",
        "Ray-Ban Meta camera privacy",
        "smart glasses privacy",
        "Meta glasses data collection",
    ],
    "KIQ_2_2": [
        "Ray-Ban Meta glasses",
        "Apple smart glasses",
        "Google Warby Parker glasses",
        "Snap Spectacles 2026",
        "Xreal glasses",
        "Xiaomi smart glasses",
    ],
}
ALL_QUERIES = [q for qs in QUERIES_BY_KIQ.values() for q in qs]

POSTS_PER_QUERY = 500
PAGE_LIMIT = 100
OUTPUT_PATH = Path("data/raw/reddit_posts.csv")


def fetch_page(params: dict, retries: int = 3) -> list[dict] | None:
    """Return a list of posts, or None on unrecoverable error."""
    for attempt in range(retries):
        try:
            r = requests.get(POSTS_ENDPOINT, params=params, timeout=30)
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


def collect_for_query(query: str) -> list[dict]:
    collected: dict[str, dict] = {}
    for sub in SUBREDDITS:
        if len(collected) >= POSTS_PER_QUERY:
            break
        after = AFTER_TS
        while len(collected) < POSTS_PER_QUERY:
            params = {
                "subreddit": sub,
                "query": query,
                "sort": "asc",
                "after": after,
                "before": BEFORE_TS,
                "limit": PAGE_LIMIT,
            }
            page = fetch_page(params)
            if page is None or not page:
                break
            for p in page:
                pid = p.get("id")
                if not pid or pid in collected:
                    continue
                collected[pid] = p
                if len(collected) >= POSTS_PER_QUERY:
                    break
            if len(page) < PAGE_LIMIT:
                break
            last_ts = page[-1].get("created_utc")
            if not isinstance(last_ts, (int, float)):
                break
            after = int(last_ts) + 1
            time.sleep(0.5)
        time.sleep(1)
    return list(collected.values())


def to_row(post: dict, query: str) -> dict:
    permalink = post.get("permalink") or ""
    url = f"https://www.reddit.com{permalink}" if permalink else ""
    ts = post.get("created_utc")
    date_str = (
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        if isinstance(ts, (int, float)) else ""
    )
    return {
        "post_id": post.get("id", ""),
        "title": post.get("title") or "",
        "text": post.get("selftext") or "",
        "score": post.get("score", 0),
        "created_utc": date_str,
        "subreddit": post.get("subreddit", ""),
        "num_comments": post.get("num_comments", 0),
        "url": url,
        "query": query,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows: list[dict] = []
    counts: dict[str, int] = {}

    for query in tqdm(ALL_QUERIES, desc="queries"):
        posts = collect_for_query(query)
        counts[query] = len(posts)
        for p in posts:
            pid = p.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            rows.append(to_row(p, query))
        time.sleep(1)

    fieldnames = ["post_id", "title", "text", "score", "created_utc",
                  "subreddit", "num_comments", "url", "query"]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} unique posts to {OUTPUT_PATH}")
    print("\nPosts collected per query (pre global dedupe):")
    for q, n in counts.items():
        print(f"  {n:>4}  {q}")


if __name__ == "__main__":
    main()
