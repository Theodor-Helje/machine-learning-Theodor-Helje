import pandas as pd
import numpy as np
from data import load_data
from sklearn.preprocessing import MultiLabelBinarizer, normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack


def get_mapping_dicts(index_as_keys=False, movies_df=None, ratings_df=None):
    if movies_df is None:
        movies_df = load_data(file='movies')
    if ratings_df is None:
        ratings_df = load_data(file='ratings')

    movies = movies_df.drop_duplicates(subset='movieId').reset_index(drop=True)
    users = ratings_df.drop_duplicates(subset='userId').reset_index(drop=True)

    if index_as_keys:
        return movies['movieId'].to_dict(), users['userId'].to_dict()
    else:
        return {id: index for index, id in enumerate(movies['movieId'].unique())}, {id: index for index, id in enumerate(users['userId'].unique())}


def get_encoded_movies(movies_df=None):
    if movies_df is None:
        movies = load_data(file='movies')

    movies['genres'] = movies['genres'].str.split('|')

    mlb = MultiLabelBinarizer()
    movies_matrix = mlb.fit_transform(movies['genres'])

    movie_sorting = movies['movieId'].map(get_mapping_dicts()[0]).argsort()

    return csr_matrix(movies_matrix[movie_sorting])


def get_tfidf_encoded_tags(tags_df=None):
    if tags_df is None:
        tags = load_data(file='tags')

    tags_grouped = tags.dropna(axis=0).groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower())
    mapping = get_mapping_dicts()[0]

    tfidf_vectorizer = TfidfVectorizer(max_features=7500, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tags_grouped)

    tags_sorting = tags_grouped.index.map(mapping)

    tags_sparse = csr_matrix(
        (tfidf_matrix.data,
         (tags_sorting[tfidf_matrix.nonzero()[0]],
         tfidf_matrix.nonzero()[1])),
         shape=(len(mapping), tfidf_matrix.shape[1])
    )

    return tags_sparse


def get_movie_features_matrix(load_file=True, genre_to_tags_ratio=0.5): # change from load_file to matrix input
    if load_file:
        return load_data(file='movie_feature_matrix')

    genre_matrix = normalize(get_encoded_movies()) * genre_to_tags_ratio
    tags_matrix = normalize(get_tfidf_encoded_tags()) * (1 - genre_to_tags_ratio)

    features = hstack([genre_matrix, tags_matrix]).tocsr()

    return features.tocsr()


def get_user_interaction_matrix(load_file=True): # change from load_file to matrix input
    if load_file:
        return load_data(file='interaction_matrix')

    ratings = pd.read_csv('Labb-1/ml-latest/ratings.csv')
    mapping = get_mapping_dicts()

    ratings['movieId'] = ratings.movieId.map(mapping[0])
    ratings['userId'] = ratings.userId.map(mapping[1])

    scaler = StandardScaler() # testing
    ratings['scaled_rating'] = scaler.fit_transform(ratings[['rating']])

    interaction_matrix = csr_matrix((
        ratings['scaled_rating'],
        (ratings['userId'], ratings['movieId'])
    ))

    return interaction_matrix