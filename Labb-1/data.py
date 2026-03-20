import pandas as pd
import os
import yaml
from scipy.sparse import save_npz, load_npz


def load_file(file="movies", path=None):
    if path is None:
        default_paths = {
            "movies": "Labb-1/data-files/ml-latest/movies.csv",
            "ratings": "Labb-1/data-files/ml-latest/ratings.csv",
            "tags": "Labb-1/data-files/ml-latest/tags.csv",
            "movie_feature_matrix": "Labb-1/data-files/matrices/movie_feature_matrix.npz",
            "interaction_matrix": "Labb-1/data-files/matrices/interaction_matrix.npz",
            "hyperparameters": "Labb-1/hyperparameters.yml"
        }
        path = default_paths[file]

    if os.path.exists(path):
        if path[-1] == "v":
            return pd.read_csv(path)
        elif path[-1] == "l":
            return yaml.safe_load(path)
        else:
             return load_npz(path)
    else:
        raise FileNotFoundError(f"file {path} not found")


def save_file(file_name, file):
    if file_name[-1] == "v":
        os.makedirs("Labb-1/data-files/csv_files", exist_ok=True)
        file.to_csv(f"Labb-1/data-files/csv_files/{file_name}")
    else:
        os.makedirs("Labb-1/data-files/matrices", exist_ok=True)
        save_npz(f"Labb-1/data-files/matrices/{file_name}", file)



if __name__ == "__main__":
     print(f"movies:\n{load_file("movies").head()}\n")
     print(f"ratings:\n{load_file("ratings").head()}\n")
     print(f"tags:\n{load_file("tags").head()}\n")