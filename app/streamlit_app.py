import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from recommender import recommend_tracks

st.set_page_config(page_title='Spotify Track Analysis', layout='wide')

st.markdown("""
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stText,
    h1, h2, h3, h4, h5, h6, p, span, label, div, button,
    input, select, textarea, .stSlider label, .stSelectbox label,
    .stTextInput label, .stMetric, .stDataFrame {
        font-family: 'Times New Roman', Times, serif !important;
    }

    .stApp {
        background-color: #ffffff;
    }

    h1 {
        font-weight: 400;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #000000;
        padding-bottom: 0.6rem;
        margin-bottom: 0.3rem;
    }

    h2, h3 {
        font-weight: 400;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #cccccc;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Times New Roman', Times, serif;
        font-size: 16px;
        color: #000000;
    }

    .stButton button {
        font-family: 'Times New Roman', Times, serif;
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #000000;
        border-radius: 0px;
        padding: 0.4rem 1.2rem;
    }

    .stButton button:hover {
        background-color: #000000;
        color: #ffffff;
        border: 1px solid #000000;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Times New Roman', Times, serif;
    }

    hr {
        border: none;
        border-top: 1px solid #cccccc;
    }

    /* Slider track and thumb, overriding Streamlit's default red */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #000000 !important;
        border-color: #000000 !important;
    }

    div[data-baseweb="slider"] > div > div > div {
        background-color: #000000 !important;
    }

    div[data-baseweb="slider"] > div > div {
        background-color: #dddddd !important;
    }

    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        font-family: 'Times New Roman', Times, serif;
        color: #000000;
    }

    /* Slider current value label above the thumb */
    div[data-baseweb="slider"] div[data-testid="stThumbValue"] {
        color: #000000 !important;
        font-family: 'Times New Roman', Times, serif;
    }

    /* Selectbox / dropdown, overriding the default dark theme */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
    }

    div[data-baseweb="select"] span {
        color: #000000 !important;
        font-family: 'Times New Roman', Times, serif;
    }

    /* Dropdown menu list when opened */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #ffffff !important;
    }

    li[role="option"] {
        color: #000000 !important;
        font-family: 'Times New Roman', Times, serif;
    }

    /* Checkbox */
    .stCheckbox svg {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/app_dataset.csv')
    feature_matrix = np.load('../models/feature_matrix.npy')
    return df, feature_matrix

@st.cache_resource
def load_models():
    popularity_model = joblib.load('../models/popularity_model.pkl')
    genre_model = joblib.load('../models/genre_model.pkl')
    genre_encoder = joblib.load('../models/genre_label_encoder.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    return popularity_model, genre_model, genre_encoder, scaler

df, feature_matrix = load_data()
popularity_model, genre_model, genre_encoder, scaler = load_models()

st.title('Spotify Track Analysis')
st.write('Predict popularity, classify genre, and get recommendations — all from Spotify audio features.')

tab1, tab2, tab3 = st.tabs(['Popularity Predictor', 'Genre Classifier', 'Song Recommender'])

FEATURES = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_min', 'time_signature', 'explicit']

with tab1:
    st.subheader('Predict Track Popularity')
    col1, col2, col3 = st.columns(3)
    with col1:
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        energy = st.slider('Energy', 0.0, 1.0, 0.5)
        loudness = st.slider('Loudness (dB)', -60.0, 0.0, -10.0)
        speechiness = st.slider('Speechiness', 0.0, 1.0, 0.1)
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.3)
    with col2:
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
        liveness = st.slider('Liveness', 0.0, 1.0, 0.2)
        valence = st.slider('Valence', 0.0, 1.0, 0.5)
        tempo = st.slider('Tempo (BPM)', 40.0, 220.0, 120.0)
    with col3:
        duration_min = st.slider('Duration (min)', 0.5, 10.0, 3.5)
        key = st.selectbox('Key', list(range(12)))
        mode = st.selectbox('Mode', [0, 1], format_func=lambda x: 'Minor' if x == 0 else 'Major')
        time_signature = st.selectbox('Time Signature', [3, 4, 5])
        explicit = st.checkbox('Explicit')

    if st.button('Predict Popularity'):
        input_df = pd.DataFrame([[danceability, energy, key, loudness, mode, speechiness,
                                   acousticness, instrumentalness, liveness, valence, tempo,
                                   duration_min, time_signature, int(explicit)]], columns=FEATURES)
        pred = popularity_model.predict(input_df)[0]
        st.metric('Predicted Popularity', f'{pred:.1f} / 100')

with tab2:
    st.subheader('Classify Genre from Audio Features')
    st.write('Pick an existing track to see genre prediction vs. actual:')
    sample_df = df.dropna(subset=['track_name']).sample(n=min(500, len(df)), random_state=42)
    track_choice = st.selectbox('Choose a track', sample_df['track_name'].unique())

    if st.button('Classify Genre'):
        row = df[df['track_name'] == track_choice].iloc[0]
        X_input = scaler.transform(row[FEATURES].values.reshape(1, -1))
        pred_encoded = genre_model.predict(X_input)[0]
        pred_genre = genre_encoder.inverse_transform([pred_encoded])[0]
        st.write(f'**Predicted genre:** {pred_genre}')
        st.write(f'**Actual genre:** {row["track_genre"]}')

with tab3:
    st.subheader('Find Similar Songs')
    track_name_input = st.text_input('Enter a track name (e.g. "Comedy")')
    top_n = st.slider('Number of recommendations', 5, 20, 10)

    if st.button('Recommend'):
        result = recommend_tracks(track_name_input, df, feature_matrix, top_n=top_n)
        if result is None:
            st.error('Track not found. Try another name.')
        else:
            st.dataframe(result)