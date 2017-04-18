import glob
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import math
from scipy.stats import variation
from sklearn.feature_extraction.text import CountVectorizer
import bs4
from bs4 import BeautifulSoup

from utils import *

year_counter = {}

max_nb_files = 1000**1000
cnt = 0
wiki_dir='../data/wikified/'
ext='.wikification.tagged.flat.html'

docs, dates = [], []

fns = list(glob.glob(wiki_dir + '*' + ext))
fns = sorted(fns)
if max_nb_files:
    fns = fns[:max_nb_files]

for i, fn in enumerate(fns):
    if i and i % 500 == 0:
        print('- parsed', i, 'files')

    tokens = []
    y = os.path.basename(fn).replace(ext, '').split('_')[0]

    with open(fn, 'r') as f:
        text = f.read()

    for node in BeautifulSoup(text, 'lxml').descendants:
        try:
            if node.name and node.name == 'a':
                id_ = '#' + node['href'].split('/')[-1]
                tokens.append(id_)
        except:
            continue

    if tokens:
        docs.append(tokens)
        dates.append(y)

def identity(x):
    return x

vectorizer = CountVectorizer(analyzer=identity)
X = vectorizer.fit_transform(docs).toarray()
full_vocab = vectorizer.get_feature_names()
link_dict = pickle.load(open('../data/link_dict.p', 'rb'))
target_categories={'person', 'writer'}

vocab = []
for entity in full_vocab:
    try:
        atts = link_dict[entity]
        if target_categories:
            if atts and target_categories.intersection(atts) == target_categories:
                vocab.append(entity)
        else:
            vocab.append(entity)
    except KeyError:
        continue

vectorizer = CountVectorizer(analyzer=identity, vocabulary=vocab)
X = vectorizer.fit_transform(docs).toarray()

cumulative = zip(X.sum(axis=0), vocab)
N = 30
year_cumul = sorted(cumulative, reverse=True)[:N]

cvs = []
for _, year in year_cumul:
    idx = vocab.index(year)
    cv = variation(X[:, idx])
    cvs.append(cv)

cvs = np.log2(cvs)

labels = [i for _, i in year_cumul]
cumul = [i for i, _ in year_cumul]

sb.plt.clf()
sb.plt.rcParams['axes.linewidth'] = 0.4
fig, ax1 = sb.plt.subplots(figsize=(7, 12))  

# first plot slices:
ax1.scatter(cvs, cumul, 100, edgecolors='none', facecolors='none')

for cum, cv, label in zip(cumul, cvs, labels):
    label = label.replace('#', '').replace('_', ' ')
    ax1.text(cv, cum, label, ha='center', va="center",
             fontdict={'family': 'Arial', 'size': (8 / cv) * 5, 'color':'lightslategrey'})

ax1.set_title('Most frequently entities in Speculum')
ax1.set_xlabel('Coefficient of Variation')
ax1.set_ylabel('Cumulative Frequency')
sb.plt.savefig('../figures/entities_mentioned.pdf', bbox_inches=0)
sb.plt.close()
