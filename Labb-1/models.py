import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def get_embeddings(user_interaction_matrix, movie_feature_matrix, collaborative_to_content_ratio=0.5, svd_n_components=128):
    """returns normalized embeddings for users and movies"""
    collaborative_svd = TruncatedSVD(n_components=svd_n_components)
    collaborative_svd.fit_transform(user_interaction_matrix)
    movie_coll_embeddings = collaborative_svd.components_.T

    content_svd = TruncatedSVD(n_components=svd_n_components)
    movie_content_embeddings = content_svd.fit_transform(movie_feature_matrix)

    movie_embeddings = np.hstack([
        collaborative_to_content_ratio * movie_coll_embeddings, 
        (1 - collaborative_to_content_ratio) * movie_content_embeddings
    ])

    user_embeddings = user_interaction_matrix @ movie_embeddings

    return normalize(user_embeddings), normalize(movie_embeddings)


def predict_user_preferences(user_interaction_matrix, user_embeddings, movie_embeddings, user_id, mapping_dicts, n_reccomendations=5):
    "returns movie ids of the top n reccomended movies based on the given user id"
    movie_mapping = mapping_dicts[1][0]
    user_mapping = mapping_dicts[0][1]
    user_index = user_mapping[user_id]

    scores = movie_embeddings @ user_embeddings[user_index]

    user_rated_movies = user_interaction_matrix[user_index].toarray().flatten()
    scores[np.where(user_rated_movies)] = -np.inf

    return [movie_mapping[i] for i in np.argsort(scores)[-n_reccomendations:][::-1]]


def reccomend_similar_movies(movie_embeddings, movie_id, mapping_dicts, n_reccomendations=5):
    "returns movie ids of the top n reccomended movies based on the given movie id"
    matrix_to_movie_mapping = mapping_dicts[1][0]
    movie_to_matrix_mapping = mapping_dicts[0][0]
    movie_index = movie_to_matrix_mapping[movie_id]

    movie_vector_embedding = movie_embeddings[movie_index]

    scores = movie_embeddings @ movie_vector_embedding

    scores[movie_index] = -np.inf

    return [matrix_to_movie_mapping[i] for i in np.argsort(scores)[-n_reccomendations:][::-1]]



if __name__ == "__main__":
    import preprocessing as pp
    from data import load_file

    ratings = load_file(file="ratings")
    movies = load_file(file="movies")
    tags = load_file(file="tags")
    encoded_movies = pp.get_encoded_movies(movies)
    tfidf_tags = pp.get_tfidf_encoded_tags(tags)

    user_interaction_matrix = pp.build_user_interaction_matrix(ratings, pp.get_mapping_dicts(movies, ratings))
    user_embeddings, movie_embeddings = get_embeddings(user_interaction_matrix, pp.build_movie_features_matrix(encoded_movies, tfidf_tags), 0.65)
    df = movies

    mapping = pp.get_mapping_dicts()

    del tfidf_tags, encoded_movies, tags, movies, ratings

    while True:
        user_id = int(input('input user ID to reccomend movies for: '))
        movie_ids = predict_user_preferences(user_interaction_matrix, user_embeddings, movie_embeddings, user_id, mapping, 10)

        print(f"\nMovie reccomendations for user {user_id}:\n{df.loc[df['movieId'].isin(movie_ids)].set_index('movieId', drop=True)[['title', 'genres']]}\n")

        if input('enter to continue\ntype "exit" to close program\n') == 'exit':
            break

    movie_ids = reccomend_similar_movies(movie_embeddings, 858, mapping, 10)
    print(f"\nreccomendations for the godfather:\n{df.loc[df['movieId'].isin(movie_ids)].set_index('movieId', drop=True)[['title', 'genres']]}")

    movie_ids = reccomend_similar_movies(movie_embeddings, 6241, mapping, 10)
    print(f"\nreccomendations for pauline at the beach:\n{df.loc[df['movieId'].isin(movie_ids)].set_index('movieId', drop=True)[['title', 'genres']]}")