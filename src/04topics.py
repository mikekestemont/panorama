#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Script that builds a topic model of Speculum's
archive and visualizes the results, both using
word clouds and diachronically. The topic model
used in non-negative matrix factorization.
"""

import glob
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import shutil
sb.set()

from time import time
import pandas as pd
from wordcloud import WordCloud
from scipy.interpolate import spline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

IGNORE = set('IN DT -LRB- -RRB-'.split())

stopwords = set(l.strip() for l in open('en_stopwords.txt', 'r'))


n_features = 5000
n_topics = 250
n_top_words = 60
sample_len = 500


def top_words(model, feature_names, n_top_words):
    FONT_PATH = os.environ.get('FONT_PATH',
                               os.path.join(os.path.dirname(__file__),
                               'Arial.ttf'))
    try:
        shutil.rmtree('../figures/clouds')
    except:
        pass
    os.mkdir('../figures/clouds')

    info = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic = np.nan_to_num(topic)
        words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        info.append(' '.join(words))

        weights = [topic[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        try:
            freqs = {wo: we for wo, we in zip(words, weights)}
            wordcloud = WordCloud(normalize_plurals=False,
                                  font_path=FONT_PATH,
                                  background_color='white',
                                  colormap='inferno_r',
                                  width=800,
                                  height=400)
            wordcloud = wordcloud.generate_from_frequencies(freqs)
            sb.plt.imsave('../figures/clouds/'+str(topic_idx) + '.tiff',
                          wordcloud, dpi=1000)
            sb.plt.axis("off")
        except:
            continue

    return info


class SpeculumSentences(object):
    def __init__(self, max_nb_files=None, sample_len=None):
        self.fns = sorted(glob.glob('../data/tagged/*.txt.conll'))
        self.max_nb_files = max_nb_files
        self.sample_len = sample_len

        if max_nb_files:
            self.fns = self.fns[:self.max_nb_files]

    def __iter__(self):
        self.publ_years = []

        for idx, fname in enumerate(self.fns):
            if idx % 1000 == 0:
                print('-> doc', idx)

            with open(fname, 'r') as f:
                y = os.path.basename(fname).split('_')[0]

                tokens = []
                for line in f.readlines():
                    line = line.strip()

                    if line:
                        comps = line.split()
                        tok, pos, ner = comps[1], comps[3], comps[4]

                        if pos not in IGNORE and tok.isalpha():
                            tok = tok.lower()
                            if tok not in stopwords:
                                tokens.append(tok)

                si, ei = 0, self.sample_len

                while ei <= len(tokens):
                    si += self.sample_len
                    ei += self.sample_len

                    ts = ' '.join(tokens[si:ei]).strip()
                    if ts:
                        self.publ_years.append(y)
                        yield ts


chunks = SpeculumSentences(max_nb_files=10000000, sample_len=sample_len)
corpus = [text for text in chunks]

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=3,
                                   max_features=n_features,
                                   stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(corpus)
print("done in %0.3fs." % (time() - t0))

t0 = time()
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5)
X = nmf.fit_transform(tfidf)
X = np.nan_to_num(X)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

years = chunks.publ_years

topic_names = ['Topic '+str(i) for i in range(n_topics)]

df = pd.DataFrame(X, columns=topic_names)
df['year'] = [int(y) for y in years]
df_year = df.groupby(['year']).mean()

plots = []

years = list(range(min(df['year']), max(df['year']) + 1))

info = top_words(nmf, tfidf_feature_names, n_top_words)

try:
    shutil.rmtree('../figures/topics')
except:
    pass
os.mkdir('../figures/topics')

for i, topic in enumerate(topic_names):
    scores = np.nan_to_num(df_year.as_matrix()[:, i])

    if np.sum(scores):
        s = info[i][:60]+'[...]'

        x_smooth = np.linspace(min(years), max(years), 200)
        y_smooth = spline(years, scores, x_smooth)

        sb.plt.clf()
        sb.plt.scatter(years, scores, alpha=0.3)
        sb.plt.plot(x_smooth, y_smooth)

        plt.title('Average diachronic proportion of ' + topic + ':\n' + s)
        plt.xlabel('Year')
        plt.ylabel('Topic proportion')
        axes = plt.gca()
        axes.set_ylim([0, 0.015])
        sb.plt.savefig('../figures/topics/' + topic + '.pdf')
