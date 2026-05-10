"""Brand-detection + time-aggregation helpers (sentiment-engine-agnostic)."""
from __future__ import annotations

import pandas as pd

BRAND_KEYWORDS: dict[str, list[str]] = {
    "meta": ["meta", "ray-ban meta", "rayban meta"],
    "apple": ["apple"],
    "google": ["google", "warby parker"],
    "snap": ["snap", "spectacles", "snapchat"],
    "xreal": ["xreal"],
    "xiaomi": ["xiaomi"],
}


def detect_brands(text: str, keywords: dict[str, list[str]] = BRAND_KEYWORDS) -> str:
    """Return a comma-separated list of brand names whose keywords appear in `text`."""
    if not isinstance(text, str) or not text:
        return ""
    lowered = text.lower()
    matches = [b for b, kws in keywords.items() if any(kw in lowered for kw in kws)]
    return ",".join(matches)


def has_any_brand(text: str, keywords: dict[str, list[str]] = BRAND_KEYWORDS) -> bool:
    if not isinstance(text, str) or not text:
        return False
    lowered = text.lower()
    return any(kw in lowered for kws in keywords.values() for kw in kws)


def explode_brands(df: pd.DataFrame, brand_col: str = "brand") -> pd.DataFrame:
    """One row per (input row, brand). Drops rows whose brand cell is empty."""
    out = df.copy()
    out[brand_col] = out[brand_col].fillna("").astype(str)
    out = out[out[brand_col].str.len() > 0]
    out = out.assign(_brand=out[brand_col].str.split(",")).explode("_brand")
    out["_brand"] = out["_brand"].str.strip()
    out = out[out["_brand"].str.len() > 0]
    out = out.drop(columns=[brand_col]).rename(columns={"_brand": brand_col})
    return out


def sentiment_distribution(df: pd.DataFrame, label_col: str = "sentiment_label") -> pd.Series:
    counts = df[label_col].value_counts()
    return counts.reindex(["positive", "neutral", "negative"], fill_value=0)


def weekly_volume_and_sentiment(
    df: pd.DataFrame,
    *,
    date_col: str = "created_utc",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """Group by ISO-week start; return n, mean_score, share_pos, share_neg."""
    if df.empty:
        return pd.DataFrame(columns=["week", "n", "mean_score", "share_pos", "share_neg"])
    out = df.copy()
    out["_dt"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["_dt"])
    out["week"] = out["_dt"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    grouped = out.groupby("week").agg(
        n=(score_col, "size"),
        mean_score=(score_col, "mean"),
        share_pos=(label_col, lambda s: (s == "positive").mean()),
        share_neg=(label_col, lambda s: (s == "negative").mean()),
    ).reset_index()
    return grouped


def monthly_volume_and_sentiment(
    df: pd.DataFrame,
    *,
    date_col: str = "created_utc",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "n", "mean_score", "share_pos", "share_neg"])
    out = df.copy()
    out["_dt"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["_dt"])
    out["month"] = out["_dt"].dt.to_period("M").astype(str)
    grouped = out.groupby("month").agg(
        n=(score_col, "size"),
        mean_score=(score_col, "mean"),
        share_pos=(label_col, lambda s: (s == "positive").mean()),
        share_neg=(label_col, lambda s: (s == "negative").mean()),
    ).reset_index()
    return grouped
