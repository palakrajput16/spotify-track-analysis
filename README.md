# Spotify Track Analysis

**Live app:** https://palakrajput16-spotify-track-analysis-appstreamlit-app-exdqfe.streamlit.app/

An end-to-end machine learning project on Spotify's audio feature dataset: predicting track popularity, classifying genre, and recommending similar songs, all wrapped in an interactive Streamlit app.

## Overview

This project explores what Spotify's audio features (danceability, energy, valence, tempo, and so on) can and can't tell us about a track. It covers three separate ML tasks on the same underlying dataset, a regression problem, a multi-class classification problem, and a content-based recommendation system, and ties all three together into one interactive app.

## Dataset

- **114,000 tracks** across **114 genres**, exactly 1,000 tracks per genre (perfectly balanced)
- 21 columns including audio features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`), plus metadata (`artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, `track_genre`)
- `popularity` ranges 0 to 100, right-skewed, mean ~33

**A key data quirk:** 24,259 `track_id`s appear more than once, because the same song is genuinely tagged under multiple genres in Spotify's own catalog. This is handled explicitly rather than ignored, see below.

## Approach

**1. Data cleaning.** Dropped a junk index column and 3 rows with missing metadata. Created two versions of the cleaned data: the full dataset (all genre tags kept, used for genre-level EDA) and a deduplicated version, one row per unique track (used for anything modeling-related, so the same audio features never appear paired with contradictory genre labels).

**2. EDA.** Explored the popularity distribution, correlation between audio features, and how features like energy, danceability, and acousticness separate across genres. The standout finding: audio features show only weak correlation with popularity individually, which turns out to matter a lot for the regression results below.

**3. Popularity prediction (regression).** Compared Linear Regression, Random Forest, and XGBoost. XGBoost performed best but the result is modest by design, not by mistake, see Results and Limitations.

**4. Genre classification.** Trained a Random Forest classifier on the deduplicated dataset, stratified train/test split across all 114 classes. Reported both top-1 and top-5 accuracy, since top-1 alone understates performance on a 114-way problem where several genres genuinely overlap in audio-feature space (e.g. `pop` vs. `dance-pop`).

**5. Recommendation system.** Content-based, using cosine similarity over standardized audio features. Chosen deliberately over collaborative filtering, since this dataset has no user listening history to build one from.

**6. Streamlit app.** Three tabs: a popularity predictor with interactive sliders, a genre classifier that compares prediction vs. actual on existing tracks, and a song recommender.

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Popularity prediction | Linear Regression | R² | 0.03 |
| Popularity prediction | Linear Regression | MAE | 16.58 |
| Popularity prediction | Random Forest | R² | 0.16 |
| Popularity prediction | Random Forest | MAE | 15.04 |
| Popularity prediction | XGBoost | R² | 0.17 |
| Popularity prediction | XGBoost | MAE | 14.87 |
| Genre classification | Random Forest | Top-1 accuracy | 35.9% |
| Genre classification | Random Forest | Top-5 accuracy | 61.4% |
| Recommendation system | Content-based (cosine similarity) | -- | Returns musically coherent matches (0.87 to 0.94 similarity range) |

For context, random-guess baseline on 114 balanced classes is ~0.9%, so the genre classifier is well above chance.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, Streamlit, matplotlib/seaborn

## Project Structure

```
spotify-track-analysis/
├── data/
│   ├── raw/                  # original dataset.csv
│   └── processed/            # cleaned + feature-engineered CSVs
├── notebooks/                # 01-06, cleaning through recommendation
├── src/
│   └── recommender.py        # reusable recommendation logic used by the app
├── models/                   # saved .pkl models, scaler, label encoder
├── app/
│   └── streamlit_app.py
├── .streamlit/
│   └── config.toml           # theme config (minimal black/white, serif)
├── reports/
│   └── figures/               # saved EDA plots
└── requirements.txt
```

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cd app
streamlit run streamlit_app.py
```

Opens at `localhost:8501`.

## Key Limitations

- **Popularity is only weakly explained by audio features alone.** The real drivers of a track's popularity, artist reputation, marketing, playlist placement, release timing, aren't captured anywhere in this dataset. The regression result reflects that honestly rather than overselling the model.
- **Some genre overlap is unavoidable.** Genres like `pop` and `dance-pop` share very similar audio-feature profiles, which caps how high top-1 accuracy can realistically go regardless of model choice.
- **The recommender has no notion of user taste.** It only measures audio-feature similarity between tracks, not listening behavior.