#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to build a word2vec model of the
wikified version of the Speculum archive.
"""

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
                                    target_categories={'person', 'writer'})
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
vocab = [i for i, _ in iterator.counter.most_common(50)]

model = gensim.models.Word2Vec.load('w2v_model')


#cognates:
print([a for a, _ in model.most_similar('#King_Arthur', topn=10**10) if a.startswith('#')][:10])
print([a for a, _ in model.most_similar('#Chrétien_de_Troyes', topn=10**10) if a.startswith('#')][:10])
print([a for a, _ in model.most_similar('#Geoffrey_Chaucer', topn=10**10) if a.startswith('#')][:10])
print([a for a, _ in model.most_similar('#Charlemagne', topn=10**10) if a.startswith('#')][:10])

print(model.most_similar(positive=['#Geoffrey_Chaucer', '#French_language'], negative=['#English_language']))
print(model.most_similar(positive=['#Geoffrey_Chaucer', '#Latin'], negative=['#English_language']))
print(model.most_similar(positive=['#Geoffrey_Chaucer', '#Italy'], negative=['#English_language']))


full_X = np.array([model[w] for w in vocab])

fig = sns.plt.figure(figsize=(16, 8))
# aesthetic interventions:
ax = fig.add_subplot(111, axisbg='white')
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 4
plt.rcParams['lines.linewidth'] = 2
# get a more pleasing color palette:
set_link_color_palette([rgb2hex(rgb) for rgb in sns.color_palette("Set2", 10)])
# run the clustering:
linkage_obj = linkage(full_X, method='ward')
# visualize the dendrogram
d = dendrogram(Z=linkage_obj, 
                     labels=[v.replace('#', '').replace('_', ' ') for v in vocab],
                     leaf_font_size=16,
                     leaf_rotation=180,
                     above_threshold_color='#AAAAAA')
# some more aesthetic interventions:
ax = sns.plt.gca()
for idx, label in enumerate(ax.get_xticklabels()):
    label.set_rotation('vertical')
    label.set_fontname('Arial')
    label.set_fontsize(16)
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


