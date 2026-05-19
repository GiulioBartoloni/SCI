"""Render a wordcloud PNG for every unigram CSV under outputs/kiq_*.

Reads `term,count` CSVs produced by kiq_2_2_bigrams.py / kiq_3_3_bigrams.py and
writes a `*_wordcloud.png` next to each one. Polarity is inferred from the
filename (`*neg*` → red, `*pos*` → teal, `all_meta` / `full` → neutral grey).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

# Project palette (matches the rest of the dashboard).
PALETTES = {
    "negative": ["#e76f51", "#d65a3b", "#b94327", "#8f3019", "#5e1f0f"],
    "positive": ["#2a9d8f", "#218778", "#176b5d", "#0f4f44", "#08332b"],
    "neutral":  ["#5b9bd5", "#4a82b8", "#3c6a99", "#2b4e73", "#1f3b55"],
}


def polarity_from_name(name: str) -> str:
    n = name.lower()
    if "neg" in n:
        return "negative"
    if "pos" in n:
        return "positive"
    return "neutral"


def make_color_func(palette: list[str]):
    def _color(word, font_size, position, orientation, random_state=None, **kwargs):
        # Bigger font → darker shade. Bin into len(palette) buckets.
        idx = min(int(font_size / 18), len(palette) - 1)
        return palette[len(palette) - 1 - idx]  # smallest = lightest -> largest = darkest
    return _color


def render_wordcloud(csv_path: Path, *, term_col: str = "term", count_col: str = "count") -> Path:
    df = pd.read_csv(csv_path)
    if term_col not in df.columns:
        # bigram CSVs use 'bigram' as the label column — skip those here.
        raise ValueError(f"{csv_path.name}: no '{term_col}' column")
    freqs = dict(zip(df[term_col].astype(str), df[count_col].astype(int)))
    polarity = polarity_from_name(csv_path.stem)
    palette = PALETTES[polarity]

    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        prefer_horizontal=0.92,
        relative_scaling=0.55,  # font size more proportional to count
        min_font_size=12,
        max_font_size=130,
        collocations=False,     # we already gave unigrams; don't auto-bigram
        color_func=make_color_func(palette),
        random_state=42,
    ).generate_from_frequencies(freqs)

    title = csv_path.stem.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc="left", color="#444")
    out_path = csv_path.with_name(csv_path.stem + "_wordcloud.png")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    pattern = re.compile(r"kiq_.*_unigrams_.*\.csv$")
    csvs = sorted(p for p in OUTPUTS.rglob("*.csv") if pattern.match(p.name))
    if not csvs:
        print(f"No unigram CSVs found under {OUTPUTS}")
        return

    print(f"Found {len(csvs)} unigram CSVs")
    for csv_path in csvs:
        try:
            out = render_wordcloud(csv_path)
            print(f"  ✓ {csv_path.relative_to(ROOT)} -> {out.name}")
        except Exception as e:
            print(f"  ✗ {csv_path.relative_to(ROOT)}: {e}")


if __name__ == "__main__":
    main()
