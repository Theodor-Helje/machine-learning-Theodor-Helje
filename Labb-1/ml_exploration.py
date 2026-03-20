import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from pipeline import get_mapping_dicts as mapping, get_movie_features_matrix as movie_matrix, get_user_interaction_matrix as user_matrix
from sklearn.preprocessing import normalize


def predict_user_preferences(user_interaction_matrix, user_embeddings, movie_embeddings, user_id, k_reccomendations=5):
    movie_mapping = mapping(index_as_keys=True)[0]
    user_mapping = mapping(index_as_keys=False)[1]
    user_index = user_mapping[user_id]

    scores = movie_embeddings @ user_embeddings[user_index]

    user_rated_movies = user_interaction_matrix[user_index].toarray().flatten()
    scores[np.where(user_rated_movies)] = -np.inf

    return [movie_mapping[i] for i in np.argsort(scores)[-k_reccomendations:][::-1]]


def get_embeddings(user_interaction_matrix, movie_feature_matrix, collaborative_weight=0.5, content_weight=0.5):
    collaborative_svd = TruncatedSVD(n_components=128)
    collaborative_svd.fit_transform(user_interaction_matrix)
    movie_coll_embeddings = collaborative_svd.components_.T

    content_svd = TruncatedSVD(n_components=128)
    movie_content_embeddings = content_svd.fit_transform(movie_feature_matrix)

    movie_embeddings = np.hstack([
        collaborative_weight * movie_coll_embeddings, 
        content_weight * movie_content_embeddings
    ])

    user_embeddings = user_interaction_matrix @ movie_embeddings

    return normalize(user_embeddings), normalize(movie_embeddings)



if __name__ == "__main__":
    user_interaction_matrix = user_matrix(load_file=True)
    user_embeddings, movie_embeddings = get_embeddings(user_interaction_matrix, movie_matrix(load_file=True), 0.65, 0.35)
    df = pd.read_csv("Labb-1/ml-latest/movies.csv")

    while True:
        user_id = int(input('input user ID to reccomend movies for: '))
        movie_ids = predict_user_preferences(user_interaction_matrix, user_embeddings, movie_embeddings, user_id, 10)

        print(f"\nMovie reccomendations for user {user_id}:\n{df.loc[df['movieId'].isin(movie_ids)].set_index('movieId', drop=True)[['title', 'genres']]}\n")

        if input('enter to continue\ntype "exit" to close program\n') == 'exit':
            break