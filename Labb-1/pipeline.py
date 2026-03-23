import pandas as pd
import data
import preprocessing as pp
import models


def model_setup(): #validate and set up matrices and hyperparameters
    """verifies saved data and recalculates if needed"""
    hyperparameters = data.load_file(file='hyperparameters')
    current_state = data.create_state_df(hyperparameters)
    same_state = data.compare_state(current_state)

    if not same_state:
        mapping = pp.get_mapping_dicts()

        movies_encoded = pp.get_encoded_movies(data.load_file(file="movies"))
        tags_encoded = pp.get_tfidf_encoded_tags(data.load_file(file="tags"), hyperparameters['tfidf_max_features'])

        movie_features = pp.build_movie_features_matrix(movies_encoded, tags_encoded, genre_to_tags_ratio=hyperparameters['genre_to_tags_ratio'])
        interaction_matrix = pp.build_user_interaction_matrix(data.load_file(file="ratings"), mapping)

        del movies_encoded, tags_encoded

        data.save_file('interaction_matrix.npz', interaction_matrix)
        data.save_file('movie_feature_matrix.npz', movie_features)

        user_embeddings, movie_embeddings = models.get_embeddings(
            interaction_matrix, 
            movie_features, 
            collaborative_to_content_ratio=hyperparameters["collaborative_to_content_ratio"],
            svd_n_components=hyperparameters["svd_n_components"]
        )

        del movie_features, interaction_matrix

        data.save_file('user_embeddings.npy', user_embeddings)
        data.save_file('movie_embeddings.npy', movie_embeddings)
        data.save_file("state.csv", data.create_state_df(hyperparameters))

        return user_embeddings, movie_embeddings

    return data.load_file(file="user_embeddings"), data.load_file(file="movie_embeddings")


def predict_user_reccomendations(user_id, user_interaction_matrix=None, user_embeddings=None, movie_embeddings=None, mapping_dicts=None, n_reccomendations=5):
    """return a DataFrame with the top n reccomentations for a given user_id"""
    if user_interaction_matrix is None:
        user_interaction_matrix = data.load_file(file='interaction_matrix')
    
    if user_embeddings is None:
        user_embeddings = data.load_file(file="user_embeddings")
    
    if movie_embeddings is None:
        movie_embeddings = data.load_file(file="movie_embeddings")
    
    if mapping_dicts is None:
        mapping_dicts = pp.get_mapping_dicts()

    movie_ids = models.predict_user_preferences(
        user_interaction_matrix, 
        user_embeddings, 
        movie_embeddings, 
        user_id, 
        mapping_dicts, 
        n_reccomendations=n_reccomendations
    )

    movies = data.load_file(file="movies")

    return movies.loc[movies['movieId'].isin(movie_ids)].set_index('movieId', drop=True)[['title', 'genres']]



if __name__ == "__main__":
    print("verifying files and setting up model")
    user_embeddings, movie_embeddings = model_setup()

    print("loading data")
    user_interaction_matrix = data.load_file(file="interaction_matrix")
    mapping = pp.get_mapping_dicts()

    while True:
        user_id = int(input('input user ID to reccomend movies for: '))
        print(f"\nMovie reccomendations for user {user_id}:\n{predict_user_reccomendations(
            user_id=user_id, 
            user_interaction_matrix=user_interaction_matrix, 
            user_embeddings=user_embeddings, 
            movie_embeddings=movie_embeddings,
            mapping_dicts=mapping
            )}\n")

        if input('enter to continue\ntype "exit" to close program\n') == 'exit':
            break