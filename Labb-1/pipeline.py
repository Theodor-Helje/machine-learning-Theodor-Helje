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
    print(tags_tfidf.head())

    tfidf_vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
    tag_vectorized_matrix = tfidf_vectorizer.fit_transform(tags_tfidf).toarray()
    tags_vectorized = pd.DataFrame(tag_vectorized_matrix, columns=tfidf_vectorizer.get_feature_names_out())
    print(tags_vectorized.head())

    return tags_vectorized


if __name__ == "__main__":
    movies_encoded = dummy_encode_movies()
    tags_encoded = tfidf_encode_tags()

    #movies_encoded.to_csv('Labb-1/ml-latest/movies_encoded.csv')
    #tags_encoded.to_csv('Labb-1/ml-latest/tags_encoded.csv')



    print(f"dummy encoded movies:\n{movies_encoded}\n")
    print(f"tfidf encoded tags:\n{tags_encoded.columns}\n")
    #print(f"movie index mapping dict:\n{get_index_dict()}\n")