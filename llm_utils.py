"""Ollama-based topic-relevance + sentiment scorer with on-disk caching.

Calls a local Ollama server (default: http://localhost:11434) running gemma4-q6.
For every Reddit post or comment we ask the model to return a JSON object with
four fields: is_relevant (bool), sentiment_label (str), sentiment_score (float),
and reason (str). Results are cached to CSV so re-runs are instant.

Used by the per-KIQ notebooks under notebooks/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4-q6:latest"
TEXT_TRUNCATE = 1500
REQUEST_TIMEOUT_S = 120
CACHE_FLUSH_EVERY = 50

KIQ_TOPICS: dict[str, str] = {
    "KIQ_3_2": (
        "Sentiment toward Meta's 'Name Tag' facial-recognition feature in their Ray-Ban Meta "
        "AI smart glasses. Relevant: discussion of Name Tag specifically, facial recognition "
        "on smart glasses, or Meta's plans/policies for face-recognition identification on "
        "their glasses. Not relevant: generic smart-glasses content without that lens, "
        "unrelated tech, off-topic threads."
    ),
    "KIQ_3_3": (
        "Privacy concerns about consumer smart glasses (any brand: Meta, Apple, Google, Snap, "
        "Xreal, Xiaomi). Relevant: surveillance, recording without consent, data collection, "
        "security risks, or regulatory / civil-society pushback specific to smart glasses. "
        "Not relevant: smart-glasses content without a privacy framing, pure technical specs, "
        "off-topic threads."
    ),
    "KIQ_2_2": (
        "Substantive discussion of consumer smart-glasses brands (Meta / Ray-Ban Meta, Apple, "
        "Google, Snap Spectacles, Xreal, Xiaomi). Relevant: opinions, comparisons, or news "
        "about these specific products. Not relevant: 'meta' / 'apple' / 'snap' used in "
        "non-brand senses (meta-analysis, apple the fruit, snap the verb), or smart-glasses "
        "content that doesn't engage with any of these brands."
    ),
}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "sentiment_label": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
        "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["is_relevant", "sentiment_label", "sentiment_score", "reason"],
}

CACHE_COLUMNS = ["id", "is_relevant", "sentiment_label", "sentiment_score", "reason"]


def build_prompt(text: str, kiq_topic: str, kind: str) -> str:
    truncated = (text or "")[:TEXT_TRUNCATE]
    return (
        f"You are analyzing a Reddit {kind} for a strategic-intelligence project on consumer smart glasses.\n\n"
        f"TOPIC OF INTEREST:\n{kiq_topic}\n\n"
        f'REDDIT {kind.upper()}:\n"""\n{truncated}\n"""\n\n'
        "Return JSON only with these fields:\n"
        f"- is_relevant (bool): Is this {kind} substantively about the topic above? "
        "False if only tangentially related, off-topic, generic, or about something else. "
        "When in doubt, prefer false.\n"
        "- sentiment_label: one of \"positive\", \"neutral\", \"negative\". "
        f"The valence of the {kind} TOWARD THE TOPIC. If is_relevant is false, return \"neutral\".\n"
        "- sentiment_score: number in [-1, 1]. -1 strongly negative, 0 neutral, +1 strongly positive.\n"
        "- reason: ≤ 25 words explaining the judgement."
    )


def query_ollama(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    schema: dict[str, Any] = RESPONSE_SCHEMA,
    url: str = OLLAMA_URL,
    timeout: int = REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.0},
        "think": False,
    }
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    content = payload.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty response from Ollama: {payload!r}")
    return json.loads(content)


def score_item(
    text: str,
    kiq_topic: str,
    kind: str,
    *,
    model: str = DEFAULT_MODEL,
    retries: int = 2,
) -> dict[str, Any]:
    """Score a single item. On failure, retry once; on second failure return a neutral / not-relevant fallback."""
    prompt = build_prompt(text, kiq_topic, kind)
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            data = query_ollama(prompt, model=model)
            return {
                "is_relevant": bool(data["is_relevant"]),
                "sentiment_label": str(data["sentiment_label"]),
                "sentiment_score": float(data["sentiment_score"]),
                "reason": str(data.get("reason", ""))[:300],
            }
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
            last_err = exc
            time.sleep(1.0 * (attempt + 1))
    print(f"[warn] scoring failed after retries: {last_err}", file=sys.stderr)
    return {
        "is_relevant": False,
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "reason": f"score_failed: {type(last_err).__name__ if last_err else 'unknown'}",
    }


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"id": str})
        for c in CACHE_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        df["is_relevant"] = df["is_relevant"].astype("boolean")
        return df[CACHE_COLUMNS].copy()
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _flush_cache(cache_path: Path, rows: list[dict[str, Any]], existing: pd.DataFrame) -> pd.DataFrame:
    if not rows:
        return existing
    new_df = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="last")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index=False)
    return combined


def score_dataframe_with_cache(
    df: pd.DataFrame,
    *,
    text_col: str,
    id_col: str,
    cache_path: Path,
    kiq: str,
    kind: str,
    model: str = DEFAULT_MODEL,
    desc: str | None = None,
) -> pd.DataFrame:
    """Score `df[text_col]` for every row, caching by `df[id_col]` to `cache_path`.

    Returns a copy of `df` with `is_relevant`, `sentiment_label`, `sentiment_score`,
    `reason` columns merged on `id_col`.
    """
    if kiq not in KIQ_TOPICS:
        raise ValueError(f"unknown KIQ '{kiq}' (have: {list(KIQ_TOPICS)})")
    topic = KIQ_TOPICS[kiq]

    cache = _load_cache(cache_path)
    cached_ids = set(cache["id"].astype(str))

    work_df = df.copy()
    work_df[id_col] = work_df[id_col].astype(str)
    work_df = work_df.drop_duplicates(subset=id_col)
    todo = work_df[~work_df[id_col].isin(cached_ids)]
    print(f"[{kiq}/{kind}] cached: {len(cached_ids)}, to score: {len(todo)} of {len(work_df)}")

    pending: list[dict[str, Any]] = []
    pbar_desc = desc or f"{kiq}/{kind}"
    # Iterate over the two columns we actually need; itertuples(name=None) returns
    # plain tuples so column names starting with "_" are not silently renamed.
    rows = todo[[id_col, text_col]].itertuples(index=False, name=None)
    for item_id, text in tqdm(rows, total=len(todo), desc=pbar_desc):
        text = text or ""
        result = score_item(text, topic, kind, model=model)
        pending.append({"id": str(item_id), **result})
        if len(pending) >= CACHE_FLUSH_EVERY:
            cache = _flush_cache(cache_path, pending, cache)
            pending.clear()
    cache = _flush_cache(cache_path, pending, cache)

    merged = work_df.merge(cache, left_on=id_col, right_on="id", how="left").drop(columns="id")
    return merged


def smoke_test(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """One round-trip against a fixed sample, used as a pre-flight check."""
    sample = (
        "Meta is adding facial recognition to their smart glasses next quarter. "
        "I think this is going to be a privacy disaster — there's no way to opt out "
        "as a bystander, and the data flows are not transparent."
    )
    return score_item(sample, KIQ_TOPICS["KIQ_3_2"], "comment", model=model)
