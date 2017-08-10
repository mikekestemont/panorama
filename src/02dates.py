#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Script to count and visualize which medieval "dates" have
been most frequently mentioned in Speculum.
"""

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

year_counter = {}

MAX = 3000000000
cnt = 0

fns = list(glob.glob('../data/tagged/*.txt.conll'))
random.shuffle(fns)

docs, dates = [], []
for filename in fns[:MAX]:
    publ_year = int(os.path.basename(filename).split('_')[0])

    cnt += 1
    if cnt >= MAX:
        break
    if cnt % 1000 == 0:
        print('-> doc', cnt)

    with open(filename, 'r') as f:
        doc = []
        for line in f.readlines():
            line = line.strip()
            if line:
                comps = line.split()
                tok, ner = comps[1], comps[4]
                if ner == 'DATE' and (len(tok) == 4 or len(tok) == 3) and tok.isdigit():
                    year = int(tok)
                    if year > 500 and year < 1500:
                        doc.append(tok)
    if doc:
        docs.append(doc)
        dates.append(publ_year)


def identity(x):
    return x

years_mentioned = [str(i) for i in range(500, 1501)]
vectorizer = CountVectorizer(analyzer=identity,
                             vocabulary=years_mentioned)
X = vectorizer.fit_transform(docs).toarray()

cumulative = zip(X.sum(axis=0), years_mentioned)
N = 25
year_cumul = sorted(cumulative, reverse=True)[:N]

cvs = []
for _, year in year_cumul:
    idx = years_mentioned.index(year)
    cv = variation(X[:, idx])
    cvs.append(cv)

labels = [i for _, i in year_cumul]
cumul = [i for i, _ in year_cumul]

cumul = np.log(cumul)
cvs = np.log(cvs)

sb.plt.clf()
sb.plt.rcParams['axes.linewidth'] = 0.4
fig, ax1 = sb.plt.subplots(figsize=(7, 14))

# first plot slices:
ax1.scatter(cvs, cumul, 100, edgecolors='none', facecolors='none')

for cum, cv, year in zip(cumul, cvs, labels):
    ax1.text(cv, cum, year, ha='center', va="center",
             fontdict={'family': 'Arial', 'size': (8 / cv) * 7,
                       'color': 'darkslategrey'})

ax1.set_title('Most frequently mentioned years in Speculum')
ax1.set_xlabel('Coefficient of Variation (log)')
ax1.set_ylabel('Cumulative Frequency (log)')
sb.plt.savefig('../figures/02years_mentioned.pdf', bbox_inches=0)
sb.plt.close()
