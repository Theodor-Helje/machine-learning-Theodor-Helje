import pandas as pd
import os
import yaml
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
import numpy as np


def get_default_paths():
    default_paths = {
            "movies": "Labb-1/data-files/ml-latest/movies.csv",
            "ratings": "Labb-1/data-files/ml-latest/ratings.csv",
            "tags": "Labb-1/data-files/ml-latest/tags.csv",
            "movie_feature_matrix": "Labb-1/data-files/matrices/movie_feature_matrix.npz",
            "interaction_matrix": "Labb-1/data-files/matrices/interaction_matrix.npz",
            "hyperparameters": "Labb-1/hyperparameters.yml",
            "user_embeddings": "Labb-1/data-files/matrices/user_embeddings.npy",
            "movie_embeddings": "Labb-1/data-files/matrices/movie_embeddings.npy"
        }
    return default_paths


def load_file(file="movies", path=None):
    if path is None:
        default_paths = get_default_paths()
        path = default_paths[file]

    if os.path.exists(path):
        if path[-1] == "v":
            return pd.read_csv(path)
        elif path[-1] == "l":
            with open(path, 'r') as file:
                hyperparameters = yaml.safe_load(file)["reccomender"]
            return hyperparameters
        elif path[-1] == "y":
            return np.load(path)
        else:
             return load_npz(path)
    else:
        raise FileNotFoundError(f"file {path} not found")


def save_file(file_name, file):
    if file_name[-1] == "v":
        os.makedirs("Labb-1/data-files/csv_files", exist_ok=True)
        file.to_csv(f"Labb-1/data-files/csv_files/{file_name}", index=False)
    elif file_name[-1] == "z":
        os.makedirs("Labb-1/data-files/matrices", exist_ok=True)
        save_npz(f"Labb-1/data-files/matrices/{file_name}", file)
    elif file_name[-1] == "y":
        os.makedirs("Labb-1/data-files/matrices", exist_ok=True)
        np.save(f"Labb-1/data-files/matrices/{file_name}", file)
    else:
        raise ValueError("files must be saved as .npz, .npy or .csv format")


def create_state_df(hyperparameters=None):
    """creates checkpoint to save the state of the model to compare to when starting model"""
    if hyperparameters is None:
        hyperparameters = load_file(file="hyperparameters")
    
    paths = get_default_paths()

    storage_space = [os.path.getsize(path) if os.path.exists(path) else 0 for path in paths.values()]

    state = pd.DataFrame(data=[*hyperparameters.values(),
                               *storage_space])
    
    return state


def compare_state(current_checkpoint=None):
    """returns True if current state is equal to stored state"""
    if current_checkpoint is None:
        current_checkpoint = create_state_df()
    
    try:
        return np.allclose(current_checkpoint.values, load_file(path="Labb-1/data-files/csv_files/state.csv").values, equal_nan=True)
    except (FileNotFoundError, ValueError):
        return False


if __name__ == "__main__":
    print(f"hyperparameters:\n{load_file("hyperparameters")}\n")
    print(f"movies:\n{load_file("movies").head()}\n")
    print(f"ratings:\n{load_file("ratings").head()}\n")
    print(f"tags:\n{load_file("tags").head()}\n")

    print("\ncreating state DF\n")
    state = create_state_df()
    print(f"checkpoint DF:\n{state}\n")

    print("checking state")
    print(f"state unchanged: {compare_state(state)}\n")

    print("saving state")
    save_file("state.csv", state)