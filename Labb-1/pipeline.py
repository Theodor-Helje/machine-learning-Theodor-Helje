import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
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
    movies_encoded = movies.join(pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_))

    return movies_encoded.drop(['genres'], axis='columns')


def tfidf_encode_tags():
    tags = pd.read_csv('Labb-1/ml-latest/tags.csv')
    tags_tfidf = tags.dropna(axis=0).groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower())

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tag_vectorized_matrix = tfidf_vectorizer.fit_transform(tags_tfidf)

    tags_vectorized = pd.DataFrame.sparse.from_spmatrix(
        tag_vectorized_matrix, 
        index=tags_tfidf.index, 
        columns=tfidf_vectorizer.get_feature_names_out()
    ).fillna(0)

    return tags_vectorized


def get_movie_features():
    movie_features = pd.concat(
        [dummy_encode_movies().set_index(['movieId']), tfidf_encode_tags()],
        axis=1
    ).fillna(0)
    
    movie_features.index = movie_features.index.map(get_mapping_dicts()[0])
    movie_features.index.name = 'movieId_mapped'

    return movie_features


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
    print('creating movies encoding')
    movies_encoded = dummy_encode_movies()

    print('creating tags tf-idf encoding')
    tags_encoded = tfidf_encode_tags()

    print('creating concatenated DataFrame')
    movie_features = get_movie_features()

    print("saving matrices")
    save_npz('Labb-1/ml-latest/interaction_matrix.npz', get_user_interaction_matrix())
    #save_npz('Labb-1/ml-latest/interaction_matrix.npz', get_user_interaction_matrix())

    print(f"dummy encoded movies:\n{movies_encoded}\n")
    print(f"tfidf encoded tags:\n{tags_encoded}\n")
    print(f"combined DataFrame:\n{movie_features}\n")
    print(f"combined DataFrame info:\n{movie_features.info()}\n")

    print("creating user interaction matrix")
    print(f"matrix shape: {get_user_interaction_matrix().shape}")