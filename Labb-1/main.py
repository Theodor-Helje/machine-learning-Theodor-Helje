import pipeline
from data import load_file
from preprocessing import get_mapping_dicts


def run_movie_reccomender(n_reccomendations):
    """runs terminal-based movie reccomender (based on movies)"""
    print("verifying files and setting up model")
    _, movie_embeddings = pipeline.model_setup()

    print("loading data")
    mapping = get_mapping_dicts()

    print("\n")
    while True:
        movie_title = input('(type "exit" to close the program)\ninput movie title to base reccomendations on: ')

        if movie_title == "exit":
            break

        print(f"\nMovie reccomendations for user {movie_title}:\n{pipeline.predict_movie_reccomendations(
            movie_title=movie_title, 
            movie_embeddings=movie_embeddings,
            mapping_dicts=mapping,
            n_reccomendations=n_reccomendations
            )}\n")


def run_user_reccomender(n_reccomendations):
    """runs terminal-based movie reccomender (based on users)"""
    print("verifying files and setting up model")
    user_embeddings, movie_embeddings = pipeline.model_setup()

    print("loading data")
    user_interaction_matrix = load_file(file="interaction_matrix")
    mapping = get_mapping_dicts()

    print("\n")
    while True:
        user_id = input('(type "exit" to close the program)\ninput user ID to reccomend movies for: ')

        if user_id == "exit":
            break

        try:
            user_id = int(user_id)

            print(f"\nMovie reccomendations for user {user_id}:\n{pipeline.predict_user_reccomendations(
            user_id=user_id, 
            user_interaction_matrix=user_interaction_matrix, 
            user_embeddings=user_embeddings, 
            movie_embeddings=movie_embeddings,
            mapping_dicts=mapping,
            n_reccomendations=n_reccomendations
            )}\n")

        except ValueError:
            print("user id must convertable to int")



if __name__ == "__main__":
    print("running movie reccomender:")
    run_movie_reccomender(10)

    print("\n\n\n\n\nrunning user reccomender:")
    run_user_reccomender(10)