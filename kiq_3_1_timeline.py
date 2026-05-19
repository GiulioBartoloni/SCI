"""KIQ 3.1 — Regulatory activity timeline.

Reads kiq_3_1/data.csv (240+ items from EDPB, GARANTE, ICO, EUR-LEX) and
writes:

  outputs/kiq_3_1/kiq_3_1_cumulative.png   — cumulative line over time
                                             with annotated smart-glasses milestones
  outputs/kiq_3_1/kiq_3_1_monthly.csv      — month-by-month item counts
  outputs/kiq_3_1/kiq_3_1_acceleration.csv — KPI: monthly rate per period

Date strategy: every row has year_imputed + month_imputed (filled even where
the original date string was missing), so we build a synthetic date with
day=15 for every row. The cumulative chart is month-precise, which is all
we need for trend reading.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "kiq_3_1" / "data.csv"
OUT_DIR = ROOT / "outputs" / "kiq_3_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Regulatory milestones — only smart-glasses-named items.
# Each: (year, month, label, (xtext_offset_pts, ytext_offset_pts))
MILESTONES = [
    (2013, 6,  "GARANTE — Google Glass letters\n(precedent)",                          (-15,  40)),
    (2021, 9,  "GARANTE — asks Facebook about smart\nglasses (1 day after launch)",    (-40,  35)),
    (2025, 11, "GARANTE — press release on Smart Glasses",                                (-90, -50)),
]

# Meta product / strategic events overlaid as dotted vertical lines.
# Dates verified via web research against primary sources (Meta newsrooms,
# TechCrunch, NYT, FB about page).
# Each: ("YYYY-MM-DD", short label, level)  level 0 = top label, 1 = lower (stagger).
META_EVENTS = [
    ("2021-09-09", "Ray-Ban Stories",    0),
    ("2023-10-17", "Ray-Ban Meta",       1),
    ("2025-09-30", "Meta + Ray-Ban",       0),
    ("2026-02-13", "Name Tag leak",      1),
]
# Labels stagger up/down in the bottom strip to avoid horizontal collisions.
STRIP_LEVEL_Y = {0: 0.65, 1: -0.65}


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH)
    df["year_imputed"] = df["year_imputed"].astype(int)
    df["month_imputed"] = df["month_imputed"].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=df["year_imputed"], month=df["month_imputed"], day=15),
        errors="coerce",
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.set_index("date")
        .resample("MS")
        .size()
        .rename("n")
        .to_frame()
        .reset_index()
    )
    monthly["cumulative"] = monthly["n"].cumsum()
    return monthly


def acceleration_table(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly rate per period — pre-acceleration baseline vs surge."""
    bins = [
        ("pre-2018",      "2000-01-01", "2017-12-31"),
        ("2018–2023",     "2018-01-01", "2023-12-31"),
        ("2024–2026",     "2024-01-01", "2026-12-31"),
    ]
    rows = []
    for label, start, end in bins:
        sl = df[(df["date"] >= start) & (df["date"] <= end)]
        if sl.empty:
            continue
        n_months = (pd.Timestamp(end) - pd.Timestamp(start)).days / 30.44
        rows.append({
            "period": label,
            "n_items": len(sl),
            "months": round(n_months, 1),
            "items_per_month": round(len(sl) / n_months, 2),
        })
    return pd.DataFrame(rows)


def plot_cumulative(monthly: pd.DataFrame, out_path: Path) -> None:
    fig, (ax, ax_strip) = plt.subplots(
        2, 1,
        figsize=(13, 6.4),
        dpi=120,
        gridspec_kw=dict(height_ratios=(5.0, 1.2), hspace=0.0),
        sharex=True,
    )

    # ---- TOP PANEL: cumulative regulatory line + smart-glasses milestones --------
    ax.fill_between(monthly["date"], 0, monthly["cumulative"],
                    color="#e76f51", alpha=0.10, zorder=1)
    ax.plot(monthly["date"], monthly["cumulative"],
            color="#b94327", lw=2.2, zorder=3)

    final_total = int(monthly["cumulative"].max())
    y_top = final_total * 1.10

    # Vertical guides for the Meta events — extend across both panels,
    # from the top of the curve down to the strip dots, so the eye-link
    # between curve and product timeline is unmistakable.
    GUIDE_KW = dict(color="#555", lw=1.0, linestyle=":", alpha=0.7, zorder=2)
    for date_str, _label, _level in META_EVENTS:
        ax.axvline(pd.Timestamp(date_str), **GUIDE_KW)

    # Regulatory smart-glasses milestones — labels with explicit offsets.
    for y, m, label, (ox, oy) in MILESTONES:
        dt = pd.Timestamp(year=y, month=m, day=15)
        row = monthly[monthly["date"] == pd.Timestamp(year=y, month=m, day=1)]
        if row.empty:
            row = monthly.iloc[(monthly["date"] - dt).abs().argsort()[:1]]
        ycum = int(row["cumulative"].iloc[0])
        ax.scatter([dt], [ycum], s=55, color="#1f3b73",
                   zorder=4, edgecolors="white", linewidths=1.2)
        ha = "left" if ox > 0 else ("right" if ox < -40 else "center")
        ax.annotate(
            label,
            xy=(dt, ycum),
            xytext=(ox, oy),
            textcoords="offset points",
            ha=ha,
            fontsize=8.6,
            color="#222",
            arrowprops=dict(arrowstyle="-", color="#1f3b73", lw=0.6, alpha=0.7),
        )

    ax.set_ylim(0, y_top)
    ax.set_ylabel("Cumulative regulatory items")
    ax.set_title(
        f"Regulatory attention is accelerating, following field developments\n"
        f"{final_total} items from EDPB, GARANTE, ICO, EUR-LEX  ·  "
        f"Meta product/feature events on strip below",
        fontsize=12,
    )
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    # No ticks at the bottom of the top panel — the strip below now hosts them.
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # ---- BOTTOM STRIP: Meta events on a horizontal timeline ---------------------
    ax_strip.axhline(0, color="#444", lw=1.0, zorder=3)
    for date_str, label, level in META_EVENTS:
        dt = pd.Timestamp(date_str)
        # Extend the same dotted guide from the top of the strip down to the dot,
        # so the line visibly continues from the curve panel into the event marker.
        ax_strip.plot([dt, dt], [1.0, 0.0], **GUIDE_KW)
        ax_strip.scatter([dt], [0], s=60, color="#1f3b73",
                         zorder=4, edgecolors="white", linewidths=1.0)
        y = STRIP_LEVEL_Y[level]
        va = "bottom" if y > 0 else "top"
        ax_strip.annotate(
            label,
            xy=(dt, 0),
            xytext=(0, 8 if y > 0 else -8),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.8,
            color="#111",
        )

    ax_strip.set_ylim(-1.0, 1.0)
    ax_strip.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_strip.spines[spine].set_visible(False)
    ax_strip.set_xlabel("")

    # Shared x-axis: pad right edge so the last label has air.
    xmin = monthly["date"].min()
    xmax = max(pd.Timestamp("2026-08-01"), monthly["date"].max())
    ax_strip.set_xlim(xmin, xmax)
    ax_strip.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax_strip.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} items from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"By authority: {df['source'].value_counts().to_dict()}")
    print()

    monthly = monthly_counts(df)
    monthly.to_csv(OUT_DIR / "kiq_3_1_monthly.csv", index=False)

    accel = acceleration_table(df)
    accel.to_csv(OUT_DIR / "kiq_3_1_acceleration.csv", index=False)
    print("Acceleration by period:")
    print(accel.to_string(index=False))
    if len(accel) >= 2:
        baseline = accel[accel["period"] == "2018–2023"]["items_per_month"].iloc[0]
        surge = accel[accel["period"] == "2024–2026"]["items_per_month"].iloc[0]
        print(f"\nMonthly-rate multiplier 2024-26 vs 2018-23: {surge / baseline:.2f}×")

    plot_cumulative(monthly, OUT_DIR / "kiq_3_1_cumulative.png")
    print(f"\nWrote {OUT_DIR}/kiq_3_1_cumulative.png")


if __name__ == "__main__":
    main()
