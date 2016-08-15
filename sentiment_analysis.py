# import urllib.request

# define URLs
# test_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
# train_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

# define local file names
test_data_file_name = 'test_data.csv'
train_data_file_name = 'train_data.csv'

# download files using urlib
# test_data_f = urllib.request.urlretrieve(test_data_url, test_data_file_name)
# train_data_f = urllib.request.urlretrieve(train_data_url, train_data_file_name)

import pandas as pd
import numpy as np

test_data_df = pd.read_csv(test_data_file_name, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_data_file_name, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment","Text"]

# print(train_data_df.shape)
# print(test_data_df.shape)

# print(train_data_df.head())
# print(test_data_df.head())
# print(train_data_df.Sentiment.value_counts())

# print(np.mean([len(s.split(" ")) for s in train_data_df.Text]))

import re, nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

#######
# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems
########

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)
