# 📊 Social Media Engagement Analysis

A modular Python pipeline for loading, cleaning, and visualising social media engagement data across platforms, post types, and time periods.

---

## 📁 Project Structure

```
social_media_analysis/
├── src/
│   └── analysis.py        # Core pipeline (load → clean → analyse → visualise)
├── outputs/               # Auto-generated charts (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/social-media-analysis.git
cd social-media-analysis
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the pipeline
```bash
python src/analysis.py social_media_engagement.csv
```

All charts are saved to the `outputs/` folder automatically.

---

## 📊 What the Pipeline Produces

| Chart | Description |
|---|---|
| `platform_share.png` | Pie chart of average engagement per platform |
| `daily_engagement_platform.png` | Grouped bar chart by day and platform |
| `posts_per_day.png` | Post volume by day of the week |
| `hourly_volume_vs_engagement.png` | Dual-axis: post count vs. avg engagement by hour |
| `post_type_distribution.png` | Box plot of engagement per post type |
| `sentiment_engagement.png` | Bar chart of engagement by sentiment |
| `correlation_heatmap.png` | Likes / shares / comments correlation matrix |
| `monthly_trend.png` | Monthly engagement trend line |
| `post_type_pies.png` | Post types by volume and by total engagement |
| `strategy_heatmap.png` | Engagement heatmap: post type × platform |
| `dashboard.png` | 4-panel executive dashboard |

---

## 📦 Dataset Format

The pipeline expects a CSV with at least these columns:

| Column | Type | Description |
|---|---|---|
| `post_id` | string/int | Unique post identifier |
| `platform` | string | e.g. `Facebook`, `Instagram`, `Twitter` |
| `post_type` | string | e.g. `image`, `video`, `text`, `carousel`, `poll` |
| `post_day` | string | Day name, e.g. `Monday` |
| `post_time` | datetime | Full timestamp |
| `likes` | int | Like count |
| `comments` | int | Comment count |
| `shares` | int | Share count |
| `sentiment_score` | string | `positive`, `neutral`, or `negative` |

Missing values and non-numeric noise are handled automatically.

---

## 🔑 Key Findings (from sample dataset)

- **Instagram** drives the highest average engagement (42.2%)
- **Polls and videos** outperform other post types
- **1 AM and 9 PM** are peak engagement hours
- **Negative sentiment** posts receive slightly higher engagement than positive ones
- **Wednesday and Friday** generate the most total interactions

---

## 🛠 Tech Stack

- **pandas** – data wrangling
- **NumPy** – numerical operations
- **Matplotlib / Seaborn** – visualisation
- **scikit-learn** – (available for ML extensions)

---

## 📄 License

MIT License — feel free to use, modify, and share.
