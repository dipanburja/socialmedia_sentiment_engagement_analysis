"""
Social Media Engagement Analysis
=================================
A modular pipeline for loading, cleaning, exploring, and visualizing
social media engagement data.
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SENTIMENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

NUMERIC_COLS = ["likes", "comments", "shares"]
CATEGORICAL_COLS = ["platform", "post_type", "post_day", "sentiment_score"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── 1. Data Loading ─────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and perform initial validation."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)
    return df


# ─── 2. Data Cleaning ────────────────────────────────────────────────────────

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters and cast to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True).replace("", "0"),
        errors="coerce",
    ).fillna(0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Coerce numeric columns
    2. Fill / impute missing values
    3. Drop duplicates
    4. Parse datetime columns
    5. Engineer derived columns
    """
    df = df.copy()

    # ── Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # ── Categorical columns – fill with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", before - len(df))

    # ── Datetime parsing
    if "post_time" in df.columns:
        df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")
        df["hour"] = df["post_time"].dt.hour
        df["month"] = df["post_time"].dt.to_period("M").astype(str)

    # ── Sentiment normalisation
    if "sentiment_score" in df.columns and df["sentiment_score"].dtype == object:
        df["sentiment_score"] = (
            df["sentiment_score"].str.lower().str.strip().map(SENTIMENT_MAP).fillna(0)
        )

    # ── Derived engagement score
    df["engagement"] = df[["likes", "comments", "shares"]].sum(axis=1)

    logger.info("Clean dataset: %d rows", len(df))
    return df


# ─── 3. Exploratory Analysis ─────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> None:
    """Print grouped summary tables."""
    for group_col, label in [("platform", "Platform"), ("post_type", "Post Type")]:
        if group_col in df.columns:
            tbl = df.groupby(group_col)[["likes", "comments", "shares"]].mean().round(2)
            print(f"\n── Average Engagement by {label} ──\n{tbl}")

    if "platform" in df.columns:
        corr = df[["likes", "shares", "comments", "engagement"]].corr()
        print(f"\n── Correlation Matrix ──\n{corr.round(2)}")

    if "sentiment_score" in df.columns:
        corr_val = df["sentiment_score"].corr(df["engagement"])
        print(f"\nSentiment ↔ Engagement correlation: {corr_val:.2f}")


def top_posts(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top-n posts by engagement."""
    cols = [c for c in ["post_id", "platform", "post_type", "engagement"] if c in df.columns]
    return df.nlargest(n, "engagement")[cols].reset_index(drop=True)


# ─── 4. Visualisations ───────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_platform_share(df: pd.DataFrame) -> None:
    """Pie chart – average engagement share per platform."""
    data = df.groupby("platform")["engagement"].mean()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Average Engagement by Platform")
    _save(fig, "platform_share")


def plot_daily_engagement(df: pd.DataFrame) -> None:
    """Grouped bar chart – average engagement by day and platform."""
    per_platform = df.groupby(["post_day", "platform"])["engagement"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=per_platform, x="post_day", y="engagement",
                hue="platform", order=DAYS_ORDER, ax=ax)
    ax.set_title("Average Engagement by Day and Platform")
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Average Engagement")
    _save(fig, "daily_engagement_platform")


def plot_posts_per_day(df: pd.DataFrame) -> None:
    """Bar chart – number of posts per day."""
    counts = df["post_day"].value_counts().reset_index()
    counts.columns = ["post_day", "num_posts"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="post_day", y="num_posts", order=DAYS_ORDER, ax=ax)
    ax.set_title("Number of Posts Per Day")
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Number of Posts")
    _save(fig, "posts_per_day")


def plot_hourly_volume(df: pd.DataFrame) -> None:
    """Dual-axis chart – post volume vs. average engagement by hour."""
    hourly = df.groupby("hour")["engagement"].agg(["mean", "count"]).reset_index()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=hourly, x="hour", y="count", color="skyblue", alpha=0.6, ax=ax1)
    ax1.set_xlabel("Hour of Day (24 h)")
    ax1.set_ylabel("Number of Posts", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")
    ax2 = ax1.twinx()
    sns.lineplot(x=ax1.get_xticks(), y=hourly["mean"], marker="o", color="darkred", ax=ax2)
    ax2.set_ylabel("Average Engagement", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax1.set_title("Volume of Posts vs. Average Engagement by Hour")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    _save(fig, "hourly_volume_vs_engagement")


def plot_post_type_distribution(df: pd.DataFrame) -> None:
    """Box plot – engagement distribution per post type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="post_type", y="engagement", hue="post_type", ax=ax)
    ax.set_title("Engagement Distribution by Post Type")
    plt.xticks(rotation=45)
    _save(fig, "post_type_distribution")


def plot_sentiment_engagement(df: pd.DataFrame) -> None:
    """Bar chart – average engagement by sentiment category."""
    # Map numeric back to labels for display
    label_map = {v: k for k, v in SENTIMENT_MAP.items()}
    plot_df = df.copy()
    plot_df["sentiment_label"] = plot_df["sentiment_score"].map(label_map).fillna("unknown")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=plot_df, x="sentiment_label", y="engagement",
                hue="sentiment_label", palette="coolwarm", capsize=0.1,
                order=["positive", "neutral", "negative"], ax=ax)
    ax.set_title("Average Engagement by Sentiment Type")
    ax.set_xlabel("Sentiment Category")
    ax.set_ylabel("Avg Engagement")
    _save(fig, "sentiment_engagement")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap – correlation between likes, shares, comments, engagement."""
    corr = df[["likes", "shares", "comments", "engagement"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation: Likes vs. Shares vs. Comments")
    _save(fig, "correlation_heatmap")


def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Line chart – monthly average engagement trend."""
    trend = df.groupby("month")["engagement"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=trend, x="month", y="engagement", marker="o", color="teal", ax=ax)
    ax.set_title("Timeline: Monthly Engagement Trend")
    plt.xticks(rotation=45)
    _save(fig, "monthly_trend")


def plot_post_type_pie(df: pd.DataFrame) -> None:
    """Two pie charts – post type by count and by total engagement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    counts = df["post_type"].value_counts()
    ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140,
            colors=plt.cm.Paired.colors)
    ax1.set_title("Post Types by Volume")

    by_eng = df.groupby("post_type")["engagement"].sum()
    ax2.pie(by_eng, labels=by_eng.index, autopct="%1.1f%%", startangle=140,
            colors=plt.cm.Set3.colors)
    ax2.set_title("Post Types by Total Engagement")

    _save(fig, "post_type_pies")


def plot_strategy_heatmap(df: pd.DataFrame) -> None:
    """Heatmap – mean engagement for each post_type × platform combination."""
    pivot = df.pivot_table(index="post_type", columns="platform",
                           values="engagement", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu",
                cbar_kws={"label": "Average Engagement"}, ax=ax)
    ax.set_title("Strategy Matrix: Engagement by Post Type & Platform", fontsize=14)
    _save(fig, "strategy_heatmap")


def plot_dashboard(df: pd.DataFrame) -> None:
    """4-panel executive dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4)

    # A – Content mix
    counts = df["post_type"].value_counts()
    axes[0, 0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                   startangle=140, colors=sns.color_palette("pastel"))
    axes[0, 0].set_title("Strategy Mix: What We Post")

    # B – Engagement by post type
    sns.barplot(ax=axes[0, 1], data=df, x="post_type", y="engagement",
                hue="post_type", palette="viridis", legend=False)
    axes[0, 1].set_title("Impact: Avg Engagement by Type")

    # C – Monthly trend
    trend = df.groupby("month")["engagement"].mean().reset_index()
    sns.lineplot(ax=axes[1, 0], data=trend, x="month", y="engagement",
                 marker="o", color="teal")
    axes[1, 0].set_title("Timeline: Monthly Engagement Trend")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # D – Top 5 posts table
    axes[1, 1].axis("off")
    axes[1, 1].text(0.1, 0.9, "🏆 Top 5 Performing Posts", fontsize=14, fontweight="bold")
    top = top_posts(df)
    for y_pos, (_, row) in zip(np.linspace(0.72, 0.30, len(top)), top.iterrows()):
        axes[1, 1].text(
            0.1, y_pos,
            f"ID: {row.get('post_id', '–')}  |  {row.get('platform', '–')}  |  Score: {int(row['engagement'])}",
            fontsize=11,
        )

    _save(fig, "dashboard")


# ─── 5. Executive Summary ────────────────────────────────────────────────────

def executive_summary(df: pd.DataFrame) -> None:
    """Print a concise executive summary to stdout."""
    print("\n" + "=" * 40)
    print("         EXECUTIVE SUMMARY")
    print("=" * 40)
    print(f"  Total Audience Interactions : {int(df['engagement'].sum()):>10,}")
    if "platform" in df.columns:
        best_platform = df.groupby("platform")["engagement"].mean().idxmax()
        print(f"  Most Effective Platform     : {best_platform:>10}")
    if "hour" in df.columns:
        best_hour = int(df.groupby("hour")["engagement"].mean().idxmax())
        print(f"  Highest Engagement Hour     : {best_hour:>9}:00")
    if "sentiment_score" in df.columns:
        corr = df["sentiment_score"].corr(df["engagement"])
        print(f"  Sentiment ↔ Engagement Corr : {corr:>10.2f}")
    print("=" * 40 + "\n")


# ─── 6. Full Pipeline ────────────────────────────────────────────────────────

def run_pipeline(filepath: str) -> pd.DataFrame:
    """
    End-to-end pipeline:
    load → clean → analyse → visualise → summarise
    """
    df = load_data(filepath)
    df = clean_data(df)
    summary_stats(df)

    plots = [
        plot_platform_share,
        plot_daily_engagement,
        plot_posts_per_day,
        plot_hourly_volume,
        plot_post_type_distribution,
        plot_sentiment_engagement,
        plot_correlation_heatmap,
        plot_monthly_trend,
        plot_post_type_pie,
        plot_strategy_heatmap,
        plot_dashboard,
    ]
    for fn in plots:
        try:
            fn(df)
        except Exception as exc:
            logger.warning("Skipped %s: %s", fn.__name__, exc)

    print("\nTop 5 Posts:\n", top_posts(df).to_string(index=False))
    executive_summary(df)
    return df


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "social_media_engagement.csv"
    run_pipeline(csv_path)
