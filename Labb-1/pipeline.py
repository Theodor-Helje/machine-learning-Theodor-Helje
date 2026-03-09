import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def get_index_dict():
    movies = pd.read_csv('Labb-1/ml-latest/movies.csv')
    return movies['movieId'].to_dict()


def dummy_encode_movies():
    movies = pd.read_csv('Labb-1/ml-latest/movies.csv')
    movies['genres'] = movies['genres'].str.split('|')

    mlb = MultiLabelBinarizer()
    movies_encoded = movies.join(pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_))

    return movies_encoded


def tfidf_encode_tags():
    tags = pd.read_csv('Labb-1/ml-latest/tags.csv')
    tags_tfidf = tags.dropna(axis=0).groupby('movieId')['tag'].apply(lambda x: ' '.join(x).lower())

    tfidf_vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
    tag_vectorized_matrix = tfidf_vectorizer.fit_transform(tags_tfidf)

    tags_vectorized = pd.DataFrame.sparse.from_spmatrix(
        tag_vectorized_matrix, index=tags_tfidf.index, 
        columns=tfidf_vectorizer.get_feature_names_out()
    ).fillna(0)

    return tags_vectorized


if __name__ == "__main__":
    movies_encoded = dummy_encode_movies()
    tags_encoded = tfidf_encode_tags()

    #movies_encoded.to_csv('Labb-1/ml-latest/movies_encoded.csv')
    #tags_encoded.to_csv('Labb-1/ml-latest/tags_encoded.csv')

    movie_features = pd.concat(
    [movies_encoded.set_index(['movieId']), tags_encoded],
    axis=1
)

    print(f"dummy encoded movies:\n{movies_encoded}\n")
    print(f"tfidf encoded tags:\n{tags_encoded}\n")
    print(f"combined DataFrame:\n{movie_features}\n")