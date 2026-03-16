import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import load_npz
from pipeline import get_mapping_dicts


def predict_user_preferences(user_embeddings, movie_embeddings, user_id, k_reccomendations=5):
    movie_mapping = get_mapping_dicts(index_as_keys=True)[0]
    user_mapping = get_mapping_dicts(index_as_keys=False)[1]
    user_index = user_mapping[user_id]

    scores = movie_embeddings @ user_embeddings[user_index]

    interaction_matrix = load_npz("Labb-1/ml-latest/interaction_matrix.npz")
    user_rated_movies = interaction_matrix[user_index].toarray().flatten()
    scores[np.where(user_rated_movies)] = -np.inf

    return [movie_mapping[i] for i in np.argsort(scores)[-k_reccomendations:][::-1]]


def get_embeddings():
    user_interaction_matrix = load_npz("Labb-1/ml-latest/interaction_matrix.npz")
    #movie_features_matrix = load_npz("Labb-1/ml-latest/movie_feature_matrix.npz")

    svd = TruncatedSVD(n_components=32)

    user_embeddings = svd.fit_transform(user_interaction_matrix)
    movie_embeddings = svd.components_.T

    return user_embeddings, movie_embeddings



if __name__ == "__main__":
    user_embeddings, movie_embeddings = get_embeddings()
    df = pd.read_csv("Labb-1/ml-latest/movies.csv")

    user_id = int(input('input user ID to reccomend movies for: '))
    movie_ids = predict_user_preferences(user_embeddings, movie_embeddings, user_id)

    print(f"\nMovie reccomendations for user {user_id}:\n{df.loc[df['movieId'].isin(movie_ids)].reset_index(drop=True)}\n")