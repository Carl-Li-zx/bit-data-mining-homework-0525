import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.datasets import fetch_20newsgroups
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import manifold

nltk.data.path.insert(0, './nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')

def remove_non_english(text):
    english_text = re.sub(r'[^a-zA-Z]+', ' ', text)
    return english_text


def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join(w for w in word_tokens if w.lower() not in stop_words)
    return filtered_text


def tf_idf(categories=None):
    if categories is not None:
        dataset = fetch_20newsgroups(categories=categories,
                                     subset="train",
                                     shuffle=True,
                                     random_state=42)
    else:
        dataset = fetch_20newsgroups(subset="train",
                                     shuffle=True,
                                     random_state=42)
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 min_df=5)
    data = dataset.data
    for index, item in tqdm(enumerate(data),total=len(data)):
        data[index] = remove_stop_words(remove_non_english(item))
    x_tfidf = vectorizer.fit_transform(data)
    return x_tfidf, dataset.target, data


def k_means(k, X):
    y_pred = KMeans(n_clusters=k, random_state=6, n_init='auto').fit_predict(X)
    return y_pred


def tsne(x):
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(x)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
