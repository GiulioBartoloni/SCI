"""Streamlit presentation dashboard — single page.

Layout:
  KIT title: "Privacy and Regulatory Risk"
  Then KIQ 3.3 → KIQ 3.2 → KIQ 3.1 stacked vertically, in that order.

Run with:
    uv run streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="Privacy and Regulatory Risk",
    layout="wide",
)


def img(path: Path, caption: str | None = None) -> None:
    if not path.exists():
        st.error(f"Missing image: {path}")
        return
    st.image(str(path), caption=caption, use_container_width=True)


# -----------------------------------------------------------------------------
# KIT title
# -----------------------------------------------------------------------------

st.title("KIT 3 - Privacy and Regulatory Risk")
st.caption("Meta AI Smart Glasses · SCI project")

st.divider()

# -----------------------------------------------------------------------------
# KIQ 3.3
# -----------------------------------------------------------------------------

st.header("KIQ 3.1 — How does privacy show up in the conversation around Meta's smart glasses?")

img(OUT / "kiq_3_3" / "kiq_3_3_stacked_bar.png",
    caption="Sentiment split — posts vs comments")

img(OUT / "kiq_3_3" / "kiq_3_3_subreddit_polarity.png",
    caption="Polarity by subreddit (top 6 by volume)")

left, right = st.columns(2)
with left:
    st.caption("Unigrams · negative slice")
    img(OUT / "kiq_3_3" / "kiq_3_3_unigrams_neg_themed.png")
with right:
    st.caption("Bigrams · negative slice")
    img(OUT / "kiq_3_3" / "kiq_3_3_bigrams_neg_themed.png")

st.divider()

# -----------------------------------------------------------------------------
# KIQ 3.2
# -----------------------------------------------------------------------------

st.header("KIQ 3.2 — Is Name Tag more likely to drive adoption or trigger backlash that damages Meta’s smart glasses brand?")

img(OUT / "kiq_3_2" / "kiq_3_2_stacked_bar.png",
    caption="Sentiment split — posts vs comments")

img(OUT / "kiq_3_2" / "kiq_3_2_strip.png",
    caption="Every item, one dot — posts (top) vs comments (bottom)")

left, right = st.columns(2)
with left:
    st.caption("Unigrams · negative slice")
    img(OUT / "kiq_3_2" / "kiq_3_2_unigrams_neg_nametag_themed.png")
with right:
    st.caption("Bigrams · negative slice")
    img(OUT / "kiq_3_2" / "kiq_3_2_bigrams_neg_nametag_themed.png")

st.divider()

# -----------------------------------------------------------------------------
# KIQ 3.1
# -----------------------------------------------------------------------------

st.header("KIQ 3.3 — Are regulators catching up?")

st.markdown(
    "> **Impact phrase.** Regulatory activity on privacy & wearables is now "
    "running at **6× the rate** it did three years ago — and every "
    "smart-glasses milestone has been followed by a regulator response "
    "within months."
)

img(OUT / "kiq_3_1" / "kiq_3_1_cumulative.png",
    caption="Cumulative regulatory items (EDPB, GARANTE, ICO, EUR-Lex) with Meta milestones on the strip")
