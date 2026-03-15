import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import save_npz, load_npz


def get_mapping_dicts(index_as_keys=False):
    movies = pd.read_csv('Labb-1/ml-latest/movies.csv').drop_duplicates(subset='movieId').reset_index(drop=True)
    users = pd.read_csv('Labb-1/ml-latest/ratings.csv').drop_duplicates(subset='userId').reset_index(drop=True)

    if index_as_keys:
        return movies['movieId'].to_dict(), users['userId'].to_dict()
    else:
        return {id: index for index, id in enumerate(movies['movieId'].unique())}, {id: index for index, id in enumerate(users['userId'].unique())}


def dummy_encode_movies():
    movies = pd.read_csv('Labb-1/ml-latest/movies.csv')
    movies['genres'] = movies['genres'].str.split('|')

    mlb = MultiLabelBinarizer()
    movies_matrix = mlb.fit_transform(movies['genres'])

    movie_sorting = movies['movieId'].map(get_mapping_dicts()[0]).argsort()

    return csr_matrix(movies_matrix[movie_sorting])


def tfidf_encode_tags():
    tags = pd.read_csv('Labb-1/ml-latest/tags.csv')
    tags_grouped = tags.dropna(axis=0).groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower())
    mapping = get_mapping_dicts()[0]

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags_grouped)

    tags_sorting = tags_grouped.index.map(mapping)

    tags_sparse = csr_matrix(
        (tfidf_matrix.data,
         (tags_sorting[tfidf_matrix.nonzero()[0]],
         tfidf_matrix.nonzero()[1])),
         shape=(len(mapping), tfidf_matrix.shape[1])
    )

    return tags_sparse


def get_movie_features():
    genre_matrix = dummy_encode_movies()
    tags_matrix = tfidf_encode_tags()

    return hstack([genre_matrix, tags_matrix]).tocsr()


def get_user_interaction_matrix():
    ratings = pd.read_csv('Labb-1/ml-latest/ratings.csv')
    mapping = get_mapping_dicts()

    ratings['movieId'] = ratings.movieId.map(mapping[0])
    ratings['userId'] = ratings.userId.map(mapping[1])

    interaction_matrix = csr_matrix((
        ratings['rating'], 
        (ratings['userId'], ratings['movieId'])
    ))

    return interaction_matrix


if __name__ == "__main__":
    print('creating movies encoded matrix')
    movies_encoded = dummy_encode_movies()

    print('creating tags tf-idf encoded matrix')
    tags_encoded = tfidf_encode_tags()

    print('creating movie features matrix')
    movie_features = get_movie_features()

    print("creating user interaction matrix")
    interaction_matrix = get_user_interaction_matrix()

    print("saving matrices\n")
    save_npz('Labb-1/ml-latest/interaction_matrix.npz', interaction_matrix)
    save_npz('Labb-1/ml-latest/movie_feature_matrix.npz', movie_features)

    print(f"dummy encoded movies matrix shape: {movies_encoded.shape}\n")
    print(f"tfidf encoded tags matrix shape: {tags_encoded.shape}\n")
    print(f"movie features matrix shape: {movie_features.shape}\n")
    print(f"interaction matrix shape: {interaction_matrix.shape}\n")