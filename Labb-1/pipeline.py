import pandas as pd
import data
import preprocessing as pp
import models


def model_setup(): #validate and set up matrices and hyperparameters
    hyperparameters = data.load_file(file='hyperparameters')
    current_state = data.create_state_df(hyperparameters)
    same_state = data.compare_state(current_state)

    if not same_state:
        mapping = pp.get_mapping_dicts()

        movies_encoded = pp.get_encoded_movies(data.load_file(file="movies"))
        tags_encoded = pp.get_tfidf_encoded_tags(data.load_file(file="tags"))

        movie_features = pp.build_movie_features_matrix(movies_encoded, tags_encoded, genre_to_tags_ratio=hyperparameters['genre_to_tags_ratio'])
        interaction_matrix = pp.build_user_interaction_matrix(data.load_file(file="ratings"), mapping)

        del movies_encoded, tags_encoded

        data.save_file('interaction_matrix.npz', interaction_matrix)
        data.save_file('movie_feature_matrix.npz', movie_features)

        user_embeddings, movie_embeddings = models.get_embeddings(
            interaction_matrix, 
            movie_features, 
            collaborative_to_content_ratio=hyperparameters["collaborative_to_content_ratio"]
        )

        del movie_features, interaction_matrix

        data.save_file('user_embeddings.npy', user_embeddings)
        data.save_file('movie_embeddings.npy', movie_embeddings)
        data.save_file("state.csv", data.create_state_df(hyperparameters))

        return user_embeddings, movie_embeddings

    return data.load_file(file="user_embeddings"), data.load_file(file="movie_embeddings")



if __name__ == "__main__":
    print("verifying files and setting up model")
    user_embeddings, movie_embeddings = model_setup()