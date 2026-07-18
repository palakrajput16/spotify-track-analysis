import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_tracks(track_name, df, feature_matrix, top_n=10):
    matches = df[df["track_name"].str.lower() == track_name.lower()]

    if matches.empty:
        return None

    idx = matches.index[0]

    query_vector = feature_matrix[idx].reshape(1, -1)

    similarities = cosine_similarity(
        query_vector,
        feature_matrix
    )[0]

    similar_indices = similarities.argsort()[::-1]

    similar_indices = [
        i for i in similar_indices
        if i != idx
    ][:top_n]

    results = df.iloc[similar_indices][
        [
            "track_name",
            "artists",
            "track_genre",
            "popularity"
        ]
    ].copy()

    results["similarity_score"] = similarities[
        similar_indices
    ]

    return results.reset_index(drop=True)