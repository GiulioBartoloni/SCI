"""
KIT 3 — Privacy & Regulatory Risk: demo Shiny dashboard.

Mock prototype written in Shiny *for Python* (not R) so the team can iterate on
the layout before committing to an R Shiny build.

Three pages, one per KIQ:
    KIQ 3.1  — regulator publication monitor (data: kiq_3_1/data.csv)
    KIQ 3.2  — Name Tag adoption vs backlash (data: outputs/kiq_3_2/*)
    KIQ 3.3  — privacy volume & tone over time (data: outputs/kiq_3_3/*)

Run:
    uv run shiny run --reload app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, reactive, render, ui

ROOT = Path(__file__).parent

# =================================================================== shared

SENTIMENT_COLOURS = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}
SOURCE_COLOURS = {
    "ICO": "#1f77b4", "GARANTE": "#2ca02c",
    "EDPB": "#ff7f0e", "EUR-LEX": "#9467bd",
}
KIT_BANNER_CSS = """
.kiq-banner { background:#0b3d91; color:white; padding:14px 18px;
    border-radius:8px; margin-bottom:14px; }
.kiq-banner h3 { margin:0 0 4px 0; font-size:18px; }
.kiq-banner p  { margin:0; font-size:14px; opacity:.9; }
.kpi-row { display:flex; gap:12px; margin-bottom:14px; flex-wrap:wrap; }
.kpi-card { flex:1 1 160px; background:#f6f8fb; border:1px solid #e3e8ef;
    border-radius:8px; padding:12px 14px; }
.kpi-value { font-size:26px; font-weight:600; color:#0b3d91; }
.kpi-label { font-size:12px; color:#5b6577; text-transform:uppercase;
    letter-spacing:.04em; margin-top:2px; }
.section-note { color:#5b6577; font-size:13px; margin-top:6px; }
"""


def kpi_card(value_id: str, label: str) -> ui.Tag:
    return ui.div(
        ui.div(ui.output_text(value_id), class_="kpi-value"),
        ui.div(label, class_="kpi-label"),
        class_="kpi-card",
    )


def banner(kiq: str, question: str) -> ui.Tag:
    return ui.div(
        ui.h3(kiq),
        ui.tags.p(question),
        class_="kiq-banner",
    )


def empty_plot(msg: str = "no data for current filters"):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="#888")
    ax.axis("off")
    return fig


# =================================================================== KIQ 3.1

K31_PATH = ROOT / "kiq_3_1" / "data.csv"
K31_TOPICS = {
    "Smart glasses / wearables": [
        "smart glass", "smart-glass", "wearable", "google glass", "body cam",
    ],
    "Facial recognition": [
        "facial recognition", "frt", "biometric recognition", "live facial",
        "riconoscimento facciale", "clearview",
    ],
    "Biometrics (other)": ["biometric", "fingerprint", "neurotech"],
    "Video surveillance / CCTV": [
        "video surveillance", "cctv", "video device", "smart camera", "doorbell",
    ],
    "AI / big data": [
        "big data", "machine learning", "artificial intelligence",
        " ai ", "ai biometrics", "ai individual", "ai consent",
    ],
}


def load_k31() -> pd.DataFrame:
    df = pd.read_csv(K31_PATH)
    df["year"] = df["year_imputed"].astype(int)
    df["month"] = df["month_imputed"].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1), errors="coerce"
    )
    name_lc = df["name"].str.lower().fillna("")
    for topic, kws in K31_TOPICS.items():
        df[f"topic::{topic}"] = name_lc.apply(lambda s, kws=kws: any(k in s for k in kws))
    return df


K31 = load_k31()
K31_SOURCES = sorted(K31["source"].unique())
K31_TOPIC_LIST = list(K31_TOPICS.keys())

# =================================================================== KIQ 3.2

K32_DIR = ROOT / "outputs" / "kiq_3_2"


def load_k32() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(K32_DIR / "kiq_3_2_scored_posts.csv")
    comments = pd.read_csv(K32_DIR / "kiq_3_2_scored_comments.csv")
    quotes = pd.read_csv(K32_DIR / "kiq_3_2_top_quotes.csv")
    posts["created_utc"] = pd.to_datetime(posts["created_utc"])
    comments["created_utc"] = pd.to_datetime(comments["created_utc"])
    quotes["created_utc"] = pd.to_datetime(quotes["created_utc"])
    posts["kind"] = "post"
    comments["kind"] = "comment"
    return posts, comments, quotes


K32_POSTS, K32_COMMENTS, K32_QUOTES = load_k32()
K32_SUBS = sorted(set(K32_POSTS["subreddit"]) | set(K32_COMMENTS["subreddit"]))
K32_LABELS = ["negative", "neutral", "positive"]

# =================================================================== KIQ 3.3

K33_DIR = ROOT / "outputs" / "kiq_3_3"


def load_k33() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    posts = pd.read_csv(K33_DIR / "kiq_3_3_scored_posts.csv")
    comments = pd.read_csv(K33_DIR / "kiq_3_3_scored_comments.csv")
    ts = pd.read_csv(K33_DIR / "kiq_3_3_timeseries.csv")
    posts["created_utc"] = pd.to_datetime(posts["created_utc"])
    comments["created_utc"] = pd.to_datetime(comments["created_utc"])
    ts["week"] = pd.to_datetime(ts["week"])
    posts["kind"] = "post"
    comments["kind"] = "comment"
    return posts, comments, ts


K33_POSTS, K33_COMMENTS, K33_TS = load_k33()
K33_SUBS = sorted(set(K33_POSTS["subreddit"]) | set(K33_COMMENTS["subreddit"]))
K33_LABELS = ["negative", "neutral", "positive"]


# ====================================================================== UI

# ---- KIQ 3.1 page ---------------------------------------------------------

k31_page = ui.nav_panel(
    "KIQ 3.1 — Regulators",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Filters"),
            ui.input_checkbox_group(
                "k31_sources", "Source", choices=K31_SOURCES, selected=K31_SOURCES,
            ),
            ui.input_slider(
                "k31_year", "Year range",
                min=int(K31["year"].min()), max=int(K31["year"].max()),
                value=(2018, int(K31["year"].max())), step=1, sep="",
            ),
            ui.input_checkbox_group(
                "k31_topics", "Topic (heuristic over doc name)",
                choices=K31_TOPIC_LIST, selected=K31_TOPIC_LIST,
            ),
            ui.input_checkbox(
                "k31_only_topic", "Restrict to docs matching ≥1 topic", value=False,
            ),
            ui.input_text("k31_kw", "Keyword in name", placeholder="e.g. meta, glass…"),
            width=300,
        ),
        banner(
            "KIQ 3.1 · Regulator publication monitor",
            "Has any EU or US regulator issued a formal enforcement action "
            "against Meta's smart-glasses data practices by December 2026?",
        ),
        ui.div(
            kpi_card("k31_kpi_total", "Documents (filtered)"),
            kpi_card("k31_kpi_recent", "In 2024–2026"),
            kpi_card("k31_kpi_sources", "Active regulators"),
            kpi_card("k31_kpi_topical", "Smart-glasses / FRT docs"),
            class_="kpi-row",
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Volume over time",
                ui.output_plot("k31_plot_timeline", height="380px"),
                ui.div(
                    "Reading note: a sustained rise across multiple regulators "
                    "is the leading indicator we expect to see before an "
                    "enforcement action lands on Meta.",
                    class_="section-note",
                ),
            ),
            ui.nav_panel(
                "Source × topic mix",
                ui.layout_columns(
                    ui.output_plot("k31_plot_source_mix", height="340px"),
                    ui.output_plot("k31_plot_topic_mix", height="340px"),
                    col_widths=(6, 6),
                ),
            ),
            ui.nav_panel(
                "Document browser",
                ui.output_data_frame("k31_table"),
            ),
        ),
    ),
)


# ---- KIQ 3.2 page ---------------------------------------------------------

k32_page = ui.nav_panel(
    "KIQ 3.2 — Name Tag",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Filters"),
            ui.input_checkbox_group(
                "k32_subs", "Subreddit", choices=K32_SUBS, selected=K32_SUBS,
            ),
            ui.input_checkbox_group(
                "k32_kinds", "Kind", choices=["post", "comment"],
                selected=["post", "comment"],
            ),
            ui.input_checkbox_group(
                "k32_labels", "Sentiment", choices=K32_LABELS, selected=K32_LABELS,
            ),
            width=300,
        ),
        banner(
            "KIQ 3.2 · Name Tag — adoption vs backlash",
            "Is Name Tag more likely to drive adoption or trigger backlash that "
            "damages Meta's smart-glasses brand?",
        ),
        ui.div(
            kpi_card("k32_kpi_n", "Items (filtered)"),
            kpi_card("k32_kpi_negshare", "Negative share"),
            kpi_card("k32_kpi_mean", "Mean sentiment score"),
            kpi_card("k32_kpi_subs", "Subreddits"),
            class_="kpi-row",
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Sentiment mix",
                ui.layout_columns(
                    ui.output_plot("k32_plot_dist", height="340px"),
                    ui.output_plot("k32_plot_subs", height="340px"),
                    col_widths=(6, 6),
                ),
                ui.div(
                    "Posts are uniformly negative (small n=21). Comments mix "
                    "skews ~80% negative across all subreddits — consistent "
                    "across r/privacy, r/technology, r/augmentedreality.",
                    class_="section-note",
                ),
            ),
            ui.nav_panel(
                "Top quotes",
                ui.markdown(
                    "Highest-signal quotes (most negative sentiment scores). "
                    "Each row's *reason* field comes from the LLM scorer."
                ),
                ui.output_data_frame("k32_quotes"),
            ),
        ),
    ),
)


# ---- KIQ 3.3 page ---------------------------------------------------------

k33_min = pd.concat([K33_POSTS["created_utc"], K33_COMMENTS["created_utc"]]).min()
k33_max = pd.concat([K33_POSTS["created_utc"], K33_COMMENTS["created_utc"]]).max()

k33_page = ui.nav_panel(
    "KIQ 3.3 — Volume & tone",
    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("Filters"),
            ui.input_checkbox_group(
                "k33_subs", "Subreddit", choices=K33_SUBS, selected=K33_SUBS,
            ),
            ui.input_checkbox_group(
                "k33_kinds", "Kind", choices=["post", "comment"],
                selected=["post", "comment"],
            ),
            ui.input_checkbox_group(
                "k33_labels", "Sentiment", choices=K33_LABELS, selected=K33_LABELS,
            ),
            ui.input_date_range(
                "k33_dates", "Date range",
                start=k33_min.date(), end=k33_max.date(),
                min=k33_min.date(), max=k33_max.date(),
            ),
            width=300,
        ),
        banner(
            "KIQ 3.3 · Privacy volume & tone over time",
            "How is investigative media coverage of Meta's data practices "
            "shaping public perception in 2026?",
        ),
        ui.div(
            kpi_card("k33_kpi_n", "Items (filtered)"),
            kpi_card("k33_kpi_negshare", "Negative share"),
            kpi_card("k33_kpi_mean", "Mean sentiment score"),
            kpi_card("k33_kpi_weeks", "Weeks with activity"),
            class_="kpi-row",
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Volume × tone over time",
                ui.output_plot("k33_plot_ts", height="400px"),
                ui.div(
                    "Bars = weekly post + comment volume. Line = weekly mean "
                    "sentiment score (right axis, range −1 → +1).",
                    class_="section-note",
                ),
            ),
            ui.nav_panel(
                "Sentiment mix",
                ui.layout_columns(
                    ui.output_plot("k33_plot_dist", height="340px"),
                    ui.output_plot("k33_plot_subs", height="340px"),
                    col_widths=(6, 6),
                ),
            ),
            ui.nav_panel(
                "Item browser",
                ui.output_data_frame("k33_table"),
            ),
        ),
    ),
)


# ---- overall ---------------------------------------------------------------

app_ui = ui.page_navbar(
    k31_page,
    k32_page,
    k33_page,
    title="KIT 3 · Privacy & Regulatory Risk — demo dashboard",
    header=ui.tags.style(KIT_BANNER_CSS),
    fillable=False,
)


# ================================================================== server

def server(input, output, session):

    # ------------------------------------------------------------- KIQ 3.1

    @reactive.calc
    def k31_filtered() -> pd.DataFrame:
        df = K31
        df = df[df["source"].isin(input.k31_sources())]
        lo, hi = input.k31_year()
        df = df[(df["year"] >= lo) & (df["year"] <= hi)]

        chosen = list(input.k31_topics())
        if chosen:
            mask = df[[f"topic::{t}" for t in chosen]].any(axis=1)
        else:
            mask = pd.Series(False, index=df.index)
        if input.k31_only_topic():
            df = df[mask]

        kw = (input.k31_kw() or "").strip().lower()
        if kw:
            df = df[df["name"].str.lower().str.contains(kw, na=False)]
        return df

    @render.text
    def k31_kpi_total():
        return f"{len(k31_filtered()):,}"

    @render.text
    def k31_kpi_recent():
        return f"{int((k31_filtered()['year'] >= 2024).sum()):,}"

    @render.text
    def k31_kpi_sources():
        return f"{k31_filtered()['source'].nunique()}"

    @render.text
    def k31_kpi_topical():
        df = k31_filtered()
        cols = ["topic::Smart glasses / wearables", "topic::Facial recognition"]
        return f"{int(df[cols].any(axis=1).sum()):,}"

    @render.plot
    def k31_plot_timeline():
        df = k31_filtered()
        if df.empty:
            return empty_plot()
        pivot = (
            df.groupby(["year", "source"]).size().unstack(fill_value=0)
              .reindex(columns=K31_SOURCES, fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(9, 4.2))
        bottom = None
        for src in pivot.columns:
            ax.bar(
                pivot.index, pivot[src], bottom=bottom, label=src,
                color=SOURCE_COLOURS.get(src, "#888"),
            )
            bottom = pivot[src] if bottom is None else bottom + pivot[src]
        ax.set_xlabel("Year")
        ax.set_ylabel("Documents")
        ax.set_title("Regulator publications per year, by source")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    @render.plot
    def k31_plot_source_mix():
        df = k31_filtered()
        counts = df["source"].value_counts().reindex(K31_SOURCES, fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(counts.index, counts.values,
                color=[SOURCE_COLOURS.get(s, "#888") for s in counts.index])
        ax.set_xlabel("Documents")
        ax.set_title("By regulator")
        ax.spines[["top", "right"]].set_visible(False)
        max_v = max(list(counts.values) + [1])
        for i, v in enumerate(counts.values):
            ax.text(v + max_v * 0.01, i, str(v), va="center", fontsize=9)
        return fig

    @render.plot
    def k31_plot_topic_mix():
        df = k31_filtered()
        counts = pd.Series(
            {t: int(df[f"topic::{t}"].sum()) for t in K31_TOPIC_LIST}
        ).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(counts.index, counts.values, color="#0b3d91")
        ax.set_xlabel("Documents")
        ax.set_title("By topic (keyword heuristic; non-exclusive)")
        ax.spines[["top", "right"]].set_visible(False)
        max_v = max(list(counts.values) + [1])
        for i, v in enumerate(counts.values):
            ax.text(v + max_v * 0.01, i, str(v), va="center", fontsize=9)
        return fig

    @render.data_frame
    def k31_table():
        df = k31_filtered().copy()
        df = df.sort_values(["year", "month"], ascending=[False, False])
        df["date"] = df["date"].dt.strftime("%Y-%m")
        df["imputed?"] = df["is_imputed"].map({True: "yes", False: ""})
        out = df[["date", "source", "name", "imputed?"]].rename(
            columns={"date": "Date", "source": "Source",
                     "name": "Document", "imputed?": "Date imputed"}
        )
        return render.DataGrid(out, height="420px", width="100%", filters=True)

    # ------------------------------------------------------------- KIQ 3.2

    @reactive.calc
    def k32_filtered() -> pd.DataFrame:
        kinds = list(input.k32_kinds())
        parts = []
        if "post" in kinds:
            parts.append(K32_POSTS[
                ["created_utc", "subreddit", "sentiment_label",
                 "sentiment_score", "kind"]
            ])
        if "comment" in kinds:
            parts.append(K32_COMMENTS[
                ["created_utc", "subreddit", "sentiment_label",
                 "sentiment_score", "kind"]
            ])
        if not parts:
            return pd.DataFrame(columns=[
                "created_utc", "subreddit", "sentiment_label",
                "sentiment_score", "kind",
            ])
        df = pd.concat(parts, ignore_index=True)
        df = df[df["subreddit"].isin(input.k32_subs())]
        df = df[df["sentiment_label"].isin(input.k32_labels())]
        return df

    @render.text
    def k32_kpi_n():
        return f"{len(k32_filtered()):,}"

    @render.text
    def k32_kpi_negshare():
        df = k32_filtered()
        if df.empty:
            return "—"
        return f"{(df['sentiment_label'] == 'negative').mean() * 100:.1f}%"

    @render.text
    def k32_kpi_mean():
        df = k32_filtered()
        if df.empty:
            return "—"
        return f"{df['sentiment_score'].mean():+.2f}"

    @render.text
    def k32_kpi_subs():
        return f"{k32_filtered()['subreddit'].nunique()}"

    @render.plot
    def k32_plot_dist():
        df = k32_filtered()
        if df.empty:
            return empty_plot()
        pivot = (
            df.groupby(["kind", "sentiment_label"]).size().unstack(fill_value=0)
              .reindex(columns=K32_LABELS, fill_value=0)
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0) * 100
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bottom = None
        for label in K32_LABELS:
            ax.bar(pivot.index, pivot[label], bottom=bottom, label=label,
                   color=SENTIMENT_COLOURS[label])
            bottom = pivot[label] if bottom is None else bottom + pivot[label]
        ax.set_ylabel("% of items")
        ax.set_ylim(0, 100)
        ax.set_title("Sentiment mix — posts vs comments")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    @render.plot
    def k32_plot_subs():
        df = k32_filtered()
        if df.empty:
            return empty_plot()
        sub_neg = (
            df.assign(neg=(df["sentiment_label"] == "negative").astype(int))
              .groupby("subreddit")
              .agg(n=("kind", "size"), neg_share=("neg", "mean"))
              .sort_values("n", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(sub_neg.index, sub_neg["neg_share"] * 100, color="#c0392b")
        for i, (n, share) in enumerate(zip(sub_neg["n"], sub_neg["neg_share"])):
            ax.text(share * 100 + 1, i, f"n={n}", va="center", fontsize=8)
        ax.set_xlabel("% negative")
        ax.set_xlim(0, 110)
        ax.set_title("Negative share by subreddit")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    @render.data_frame
    def k32_quotes():
        df = K32_QUOTES.copy().sort_values("sentiment_score")
        df["date"] = df["created_utc"].dt.strftime("%Y-%m-%d")
        out = df[["date", "subreddit", "kind", "sentiment_score",
                  "reason", "text"]].rename(columns={
            "date": "Date", "subreddit": "Subreddit", "kind": "Kind",
            "sentiment_score": "Score", "reason": "LLM reason", "text": "Text",
        })
        return render.DataGrid(out, height="420px", width="100%", filters=True)

    # ------------------------------------------------------------- KIQ 3.3

    @reactive.calc
    def k33_filtered() -> pd.DataFrame:
        kinds = list(input.k33_kinds())
        parts = []
        cols = ["created_utc", "subreddit", "sentiment_label",
                "sentiment_score", "kind"]
        if "post" in kinds:
            parts.append(K33_POSTS[cols])
        if "comment" in kinds:
            parts.append(K33_COMMENTS[cols])
        if not parts:
            return pd.DataFrame(columns=cols)
        df = pd.concat(parts, ignore_index=True)
        df = df[df["subreddit"].isin(input.k33_subs())]
        df = df[df["sentiment_label"].isin(input.k33_labels())]
        d0, d1 = input.k33_dates()
        df = df[(df["created_utc"] >= pd.Timestamp(d0)) &
                (df["created_utc"] <= pd.Timestamp(d1) + pd.Timedelta(days=1))]
        return df

    @render.text
    def k33_kpi_n():
        return f"{len(k33_filtered()):,}"

    @render.text
    def k33_kpi_negshare():
        df = k33_filtered()
        if df.empty:
            return "—"
        return f"{(df['sentiment_label'] == 'negative').mean() * 100:.1f}%"

    @render.text
    def k33_kpi_mean():
        df = k33_filtered()
        if df.empty:
            return "—"
        return f"{df['sentiment_score'].mean():+.2f}"

    @render.text
    def k33_kpi_weeks():
        df = k33_filtered()
        if df.empty:
            return "0"
        weeks = df["created_utc"].dt.to_period("W").nunique()
        return f"{weeks}"

    @render.plot
    def k33_plot_ts():
        df = k33_filtered()
        if df.empty:
            return empty_plot()
        df = df.assign(week=df["created_utc"].dt.to_period("W").dt.start_time)
        weekly = df.groupby("week").agg(
            n=("kind", "size"), mean_score=("sentiment_score", "mean")
        ).sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.4))
        ax.bar(weekly.index, weekly["n"], width=5, color="#0b3d91",
               alpha=0.65, label="items / week")
        ax.set_ylabel("items / week")
        ax.set_xlabel("Week")
        ax.spines[["top"]].set_visible(False)
        ax2 = ax.twinx()
        ax2.plot(weekly.index, weekly["mean_score"], color="#c0392b",
                 marker="o", linewidth=2, label="mean sentiment")
        ax2.axhline(0, color="#888", linestyle=":", linewidth=0.8)
        ax2.set_ylabel("mean sentiment score")
        ax2.set_ylim(-1, 1)
        ax2.spines[["top"]].set_visible(False)
        fig.autofmt_xdate()
        ax.set_title("Weekly volume + tone")
        return fig

    @render.plot
    def k33_plot_dist():
        df = k33_filtered()
        if df.empty:
            return empty_plot()
        pivot = (
            df.groupby(["kind", "sentiment_label"]).size().unstack(fill_value=0)
              .reindex(columns=K33_LABELS, fill_value=0)
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0) * 100
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bottom = None
        for label in K33_LABELS:
            ax.bar(pivot.index, pivot[label], bottom=bottom, label=label,
                   color=SENTIMENT_COLOURS[label])
            bottom = pivot[label] if bottom is None else bottom + pivot[label]
        ax.set_ylabel("% of items")
        ax.set_ylim(0, 100)
        ax.set_title("Sentiment mix — posts vs comments")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    @render.plot
    def k33_plot_subs():
        df = k33_filtered()
        if df.empty:
            return empty_plot()
        agg = (
            df.assign(neg=(df["sentiment_label"] == "negative").astype(int))
              .groupby("subreddit")
              .agg(n=("kind", "size"), neg_share=("neg", "mean"))
              .sort_values("n", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.barh(agg.index, agg["n"], color="#0b3d91")
        for i, (n, share) in enumerate(zip(agg["n"], agg["neg_share"])):
            ax.text(n + max(agg["n"]) * 0.01, i,
                    f"{share * 100:.0f}% neg", va="center", fontsize=8)
        ax.set_xlabel("Items")
        ax.set_title("Volume by subreddit (annotated with % negative)")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    @render.data_frame
    def k33_table():
        df = k33_filtered().copy()
        if df.empty:
            return render.DataGrid(df, height="420px", width="100%")
        df = df.sort_values("created_utc", ascending=False)
        df["date"] = df["created_utc"].dt.strftime("%Y-%m-%d")
        out = df[["date", "subreddit", "kind", "sentiment_label",
                  "sentiment_score"]].rename(columns={
            "date": "Date", "subreddit": "Subreddit", "kind": "Kind",
            "sentiment_label": "Label", "sentiment_score": "Score",
        })
        return render.DataGrid(out, height="420px", width="100%", filters=True)


app = App(app_ui, server)
