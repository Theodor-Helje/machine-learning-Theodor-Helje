import pandas as pd
import os
from scipy.sparse import save_npz, load_npz


def load_file(file="movies", path=None):
    if path is None:
        default_paths = {
            "movies": "Labb-1/data-files/ml-latest/movies.csv",
            "ratings": "Labb-1/data-files/ml-latest/ratings.csv",
            "tags": "Labb-1/data-files/ml-latest/tags.csv",
            "movie_feature_matrix": "Labb-1/data-files/ml-latest/movie_feature_matrix.npz",
            "interaction_matrix": "Labb-1/data-files/ml-latest/interaction_matrix.npz"
        }
        path = default_paths[file]

    if os.path.exists(path):
        if path[-1] == "v":
            return pd.read_csv(path)
        else:
             return load_npz(path)
    else:
        raise FileNotFoundError(f"file {path} not found")


def save_file(path, file):
    if path[-1] == "v":
        file.to_csv(path)
    else:
        save_npz(path, file)


def validate_files(recalculate_matrices=False): # restructure or remove, currently compleately wrong
    if os.path.exists('Labb-1/ml-latest'):
        if not os.path.exists('Labb-1/data-files/ml-latest/movies.csv'):
            raise FileNotFoundError("Labb-1/data-files/ml-latest/movies.csv not found")
        
        elif not os.path.exists('Labb-1/data-files/ml-latest/ratings.csv'):
            raise FileNotFoundError("Labb-1/data-files/ml-latest/ratings.csv not found")
        
        elif not os.path.exists('Labb-1/data-files/ml-latest/tags.csv'):
            raise FileNotFoundError("Labb-1/data-files/ml-latest/tags.csv not found")
    else:
        raise FileNotFoundError("Labb-1/data-files/ml-latest directory not found")
    
    if not os.path.exists('Labb-1/data-files/ml-latest/interaction_matrix.npz') or recalculate_matrices:
            save_npz('Labb-1/data-files/ml-latest/interaction_matrix.npz', get_user_interaction_matrix(load_file=False))

    elif not os.path.exists('Labb-1/data-files/ml-latest/movie_feature_matrix.npz') or recalculate_matrices:
            save_npz('Labb-1/data-files/ml-latest/movie_feature_matrix.npz', get_movie_features_matrix(load_file=False))



if __name__ == "__main__":
     print(f"movies:\n{load_file("movies").head()}\n")
     print(f"ratings:\n{load_file("ratings").head()}\n")
     print(f"tags:\n{load_file("tags").head()}\n")