# SCI — Reddit Scraping Pipeline

Reddit data collection for the Meta AI Smart Glasses Strategic & Competitive Intelligence project.

KIQs covered:
- **KIQ 3.2** — Sentiment on Meta's Name Tag facial recognition feature
- **KIQ 3.3** — Volume and tone of privacy-related posts over time
- **KIQ 2.2** — Brand mention share across subreddits

## Source

Posts and comments are pulled from the [Arctic Shift](https://arctic-shift.photon-reddit.com/) public mirror (the original Pushshift API is no longer available to non-moderators).

Arctic Shift's keyword search requires a `subreddit` (or `author`) filter, so each query is run across a curated list of tech / privacy / AR / brand subreddits defined in [collect_posts.py](collect_posts.py).

## Setup

```sh
uv sync
```

## Run

```sh
uv run collect_posts.py     # -> data/raw/reddit_posts.csv
uv run collect_comments.py  # -> data/raw/reddit_comments.csv
uv run preprocess.py        # -> data/processed/reddit_*_clean.csv
uv run label_kiq.py         # adds kiq_label and brand columns
```

## Output

```
data/
  raw/
    reddit_posts.csv
    reddit_comments.csv
  processed/
    reddit_posts_clean.csv
    reddit_comments_clean.csv
```
