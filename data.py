import pandas as pd
import os


def load_data(file="movies", path=None):
    if path is None:
        default_paths = {
            "movies": "Labb-1/ml-latest/movies.csv",
            "ratings": "Labb-1/ml-latest/ratings.csv",
            "tags": "Labb-1/ml-latest/tags.csv",
            "movie_feature_matrix": "Labb-1/ml-latest/movie_feature_matrix.npz",
            "interaction_matrix": "Labb-1/ml-latest/interaction_matrix.npz"
        }
        path = default_paths[file]

    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"file {path} not found")