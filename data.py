import pandas as pd
import nltk.data
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
import re
import json


class Data:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.data_uncleaned = []
        self.data_tokenized = []
        self.labels = []
        self.stemmed_docs = []
        self.ngrams = []
        self.char_ngrams = []

    def read_txt_data(self):
        lines = []

        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        with open(self.data_dir) as f:
            for line in f.readlines():
                lines.extend(tokenizer.tokenize(line))

        self.data = lines

    def set_data(self):
        data = pd.read_csv(self.data_dir)
        self.data = data.TITLE[:20]
        self.labels = data.CATEGORY

    def build_model(self, stem=True, n=2):
        dirty_docs = []
        cleaned_ngrams = []

        cleaned_tokenized_docs = []
        stop_words = set(stopwords.words('english'))

        for document in self.data:
            document = re.sub(r"[^a-z\d ]", "", document.lower())
            dirty_docs.append(document)
            tokens = word_tokenize(document)
            cleaned_vec = [w for w in tokens if w.lower() not in stop_words]
            cleaned_tokenized_docs.append(cleaned_vec)
            cleaned_ngrams.append(list(ngrams(cleaned_vec, n)))

        self.data_tokenized = list(
            filter(lambda x: len(x) > 0, cleaned_tokenized_docs))

        self.data_uncleaned = dirty_docs
        self.ngrams = cleaned_ngrams
        self.char_ngrams = list(ngrams("".join(self.data).split(), n))
        stemmed = []

        if stem:
            ps = PorterStemmer()

            for doc in self.data_tokenized:
                stem_doc = []

                for word in doc:
                    stem_doc.append(ps.stem(word))

                stemmed.append(stem_doc)

            self.stemmed_docs = stemmed

    def generate_ngrams(self):
        ngrs = []

        for doc in self.data:
            ngrs.append(list(ngrams(doc, 3)))

        return list(ngrs)
