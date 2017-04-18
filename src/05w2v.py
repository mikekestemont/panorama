#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import glob
import os
import pickle
import random
from collections import Counter

from lxml import etree

import bs4
from bs4 import BeautifulSoup

import gensim
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.colors import rgb2hex
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette

from utils import *

SEED = 864164
random.seed(SEED)

punct = set(". ? , | `` '' ' : \" \"\" ; § % ".split())

# - top 10 of authors, kings, poems
# - table with closest cognates, #Holy_Grail
# - trees for target category poem, book, saint, order


# saints!!! books!!!



M = 2000000
print('parsing wiki xml...')
#parse_wikified(max_nb_files=M)


# text  source book
# person writer
# monastery

# manuscripts closest to libraries?
# character in interesting!

iterator = WikifiedSpeculumSentences(max_nb_files=M,
                                    #target_categories={'person', 'writer'},
                                    target_categories={})
                                    #target_categories={'monastery'})
print('running iterator...')
sentences = [s for s in iterator]

"""
print('running w2v...')
model = gensim.models.Word2Vec(sentences,
                               size=300,
                               min_count=10,
                               window=10,
                               seed=864164)

model.save('w2v_model')
"""



print(iterator.counter.most_common(100))

model = gensim.models.Word2Vec.load('w2v_model')


#cognates:
#print('grail', model.most_similar('#Holy_Grail', topn=100))
#print('Virgil ->', model.most_similar('#Virgil', topn=100))
#print([a for a, _ in model.most_similar('#King_Arthur', topn=10**10) if a.startswith('#')][:10])
#print([a for a, _ in model.most_similar('#Chrétien_de_Troyes', topn=10**10) if a.startswith('#')][:10])
#print([a for a, _ in model.most_similar('#Geoffrey_Chaucer', topn=10**10) if a.startswith('#')][:10])
#print([a for a, _ in model.most_similar('#Charlemagne', topn=10**10) if a.startswith('#')][:10])
#print([a for a, _ in model.most_similar('#Hildegard_of_Bingen', topn=10**10) if a.startswith('#')][:10])


# PROTO-TYPICALITY:
#print('PROTO-TYPICALITY >>>>>')
#vocab = set([i for i,_ in iterator.counter.most_common()])
#print(vocab)
#print([a for a, _ in model.most_similar(positive=['literature'], topn=10**10) \
#            if a.startswith('#') and a in vocab][:10])

#print(model.most_similar(positive=['men', 'man']))
#print(model.most_similar(positive=['woman', 'women']))

#print(model.most_similar(positive=['#Geoffrey_Chaucer', '#French_language'], negative=['#English_language']))
#print(model.most_similar(positive=['#Geoffrey_Chaucer', '#Latin'], negative=['#English_language']))
#print(model.most_similar(positive=['#Geoffrey_Chaucer', '#Germany'], negative=['#English_language']))
#print(model.most_similar(positive=['#Geoffrey_Chaucer', '#Italy'], negative=['#English_language']))

vocab = [i for i,_ in model.most_similar('#Holy_Grail', topn=1000000) if i.startswith('#')][:250]
print(vocab)
full_X = np.array([model[w] for w in vocab])

fig = sns.plt.figure()
# aesthetic interventions:
ax = fig.add_subplot(111, axisbg='white')
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 4
plt.rcParams['lines.linewidth'] = 0.75
# get a more pleasing color palette:
set_link_color_palette([rgb2hex(rgb) for rgb in sns.color_palette("Set2", 10)])
# run the clustering:
linkage_obj = linkage(full_X, method='ward')
# visualize the dendrogram
d = dendrogram(Z=linkage_obj, 
                     labels=[v.replace('#', '').replace('_', ' ') for v in vocab],
                     leaf_font_size=5,
                     leaf_rotation=180,
                     above_threshold_color='#AAAAAA')
# some more aesthetic interventions:
ax = sns.plt.gca()
for idx, label in enumerate(ax.get_xticklabels()):
    label.set_rotation('vertical')
    label.set_fontname('Arial')
    label.set_fontsize(5)
ax.get_yaxis().set_ticks([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
sns.plt.xticks(rotation=90)
sns.plt.tick_params(axis='x', which='both', bottom='off', top='off')
sns.plt.tick_params(axis='y', which='both', bottom='off', top='off')
ax.xaxis.grid(False)
ax.yaxis.grid(False)
sns.plt.rcParams["figure.facecolor"] = "white"
sns.plt.rcParams["axes.facecolor"] = "white"
sns.plt.rcParams["savefig.facecolor"] = "white"
sns.plt.subplots_adjust(bottom=0.15)
sns.plt.tight_layout()
fig.savefig('../figures/tree.pdf')

sns.plt.clf()
fig, ax1 = sns.plt.subplots(figsize=(12, 12))

pca_X = PCA(n_components=25).fit_transform(full_X)
X = TSNE(n_components=2,
                random_state=1987,
                verbose=1,
                n_iter=10000,
                perplexity=40.0,
                early_exaggeration=40.0,
                learning_rate=10).fit_transform(pca_X)

nb_clusters = 8
cl = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
clusters = cl.fit_predict(X)
colors = sns.color_palette('husl', n_colors=nb_clusters)
colors = [tuple([int(c * 256) for c in color]) for color in colors]
colors = ['#%02x%02x%02x' % colors[i] for i in clusters]

x1, x2 = X[:,0], X[:,1]
ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')
clustering = AgglomerativeClustering(linkage='ward',
                    affinity='euclidean', n_clusters=nb_clusters)
clustering.fit(X)
labels = [n.replace('#', '').replace('_', '\n') for n in vocab]
for x, y, name, cluster_label in zip(x1, x2, labels, clustering.labels_):
    ax1.text(x, y, name, ha='center', va="center",
             color=plt.cm.spectral(cluster_label / 10.),
             fontdict={'family': 'Arial', 'size': 8})
# control aesthetics:
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_yticklabels([])
ax1.set_yticks([])
sns.plt.savefig('tsne.pdf', bbox_inches=0)
sns.plt.close()

