import pandas as pd
from data import load_file, save_file
from sklearn.preprocessing import MultiLabelBinarizer, normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack


def get_mapping_dicts(movies_df=None, ratings_df=None):
    """mapping[0][] = data set Id as keys\n
    mapping[1][] = matrix index as keys\n
    mapping[][0] = movies\n
    mapping[][1] = users\n"""

    if movies_df is None:
        movies_df = load_file(file='movies')
    if ratings_df is None:
        ratings_df = load_file(file='ratings')

    movies = movies_df.drop_duplicates(subset='movieId').reset_index(drop=True)
    users = ratings_df.drop_duplicates(subset='userId').reset_index(drop=True)

    return [[{id: index for index, id in enumerate(movies['movieId'].unique())}, {id: index for index, id in enumerate(users['userId'].unique())}], # no index as keys
            [movies['movieId'].to_dict(), users['userId'].to_dict()]] # index as keys


def get_encoded_movies(movies_df=None):
    if movies_df is None:
        movies_df = load_file(file='movies')

    movies_df['genres'] = movies_df['genres'].str.split('|')

    mlb = MultiLabelBinarizer()
    movies_matrix = mlb.fit_transform(movies_df['genres'])

    movie_sorting = movies_df['movieId'].map(get_mapping_dicts()[0][0]).argsort()

    return csr_matrix(movies_matrix[movie_sorting])


def get_tfidf_encoded_tags(tags_df=None, tfidf_max_features=128):
    if tags_df is None:
        tags_df = load_file(file='tags')

    tags_grouped = tags_df.dropna(axis=0).groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower())
    mapping = get_mapping_dicts()[0][0]

    tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags_grouped)

    tags_sorting = tags_grouped.index.map(mapping)

    tags_sparse = csr_matrix(
        (tfidf_matrix.data,
         (tags_sorting[tfidf_matrix.nonzero()[0]],
         tfidf_matrix.nonzero()[1])),
         shape=(len(mapping), tfidf_matrix.shape[1])
    )

    return tags_sparse


def build_movie_features_matrix(encoded_movies, tfidf_encoded_tags, genre_to_tags_ratio=0.5):
    genre_matrix = normalize(encoded_movies) * genre_to_tags_ratio
    tags_matrix = normalize(tfidf_encoded_tags) * (1 - genre_to_tags_ratio)

    features = hstack([genre_matrix, tags_matrix]).tocsr()

    return features.tocsr()


def build_user_interaction_matrix(ratings_df=None, mapping_dicts=None):
    if ratings_df is None:
        ratings_df = load_file(file='ratings')
    if mapping_dicts is None:
        mapping_dicts = get_mapping_dicts()[0]

    ratings_df['movieId'] = ratings_df.movieId.map(mapping_dicts[0][0])
    ratings_df['userId'] = ratings_df.userId.map(mapping_dicts[0][1])

    scaler = StandardScaler()
    ratings_df['scaled_rating'] = scaler.fit_transform(ratings_df[['rating']])

    interaction_matrix = csr_matrix((
        ratings_df['scaled_rating'],
        (ratings_df['userId'], ratings_df['movieId'])
    ))

    return interaction_matrix



if __name__ == "__main__":
    print('creating movies encoded matrix')
    movies_encoded = get_encoded_movies(load_file(file="movies"))

    print('creating tags tf-idf encoded matrix')
    tags_encoded = get_tfidf_encoded_tags(load_file(file="tags"))

    print('creating movie features matrix')
    movie_features = build_movie_features_matrix(movies_encoded, tags_encoded, genre_to_tags_ratio=0.5)

    print("creating user interaction matrix")
    interaction_matrix = build_user_interaction_matrix(load_file(file="ratings"), get_mapping_dicts())

    print("saving matrices\n")
    save_file('interaction_matrix.npz', interaction_matrix)
    save_file('movie_feature_matrix.npz', movie_features)

    print(f"dummy encoded movies matrix shape: {movies_encoded.shape}\n")
    print(f"tfidf encoded tags matrix shape: {tags_encoded.shape}\n")
    print(f"movie features matrix shape: {movie_features.shape}\n")
    print(f"interaction matrix shape: {interaction_matrix.shape}\n")