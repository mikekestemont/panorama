import glob
import os
from operator import itemgetter
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import pandas as pd
sb.set()

from scipy.stats import kendalltau

ALLOW = set('NN NNS'.split())

stopwords = set(l.strip() for l in open('en_stopwords.txt', 'r'))

fns = sorted(glob.glob('../data/tagged/*.txt.conll'))
max_nb_files = 2000000
fns = fns[:max_nb_files]

counters = {}

for idx, fname in enumerate(fns):
    if idx % 1000 == 0:
        print('-> doc', idx)

    with open(fname, 'r') as f:
        year = os.path.basename(fname).split('_')[0]
        if year not in counters:
            counters[year] = Counter()

        tokens = []
        for line in f.readlines():
            line = line.strip()
            if line:
                comps = line.split()
                tok, pos, ner = comps[1], comps[3], comps[4]
                if pos in ALLOW and tok.isalpha():
                    tok = tok.lower()
                    if tok not in stopwords:
                        tokens.append(tok)

        counters[year].update(tokens)

vocab = set()
for counter in counters:
    vocab.update(counters[counter].keys())
vocab = tuple(sorted(vocab))
print(len(vocab))

X = np.zeros((len(counters), len(vocab)))

for idx_y, year in enumerate(sorted(counters)):
    sum_ = sum(counters[year].values())
    for idx_w, w in enumerate(vocab):
        try:
            X[idx_y, idx_w] = counters[year][w] / sum_
        except:
            pass

print(X.shape)

df = pd.DataFrame(X)
df.columns = vocab
df.index = sorted(counters.keys())
scores = []

ranks = range(1,len(df.index)+1)
for feat in df.columns:
    tau, p = kendalltau(ranks, df[feat].tolist())
    scores.append((feat, tau))
scores.sort(key=itemgetter(1))
nb = 5
top, bottom = scores[:nb], scores[-nb:]
fig = sb.plt.figure()
sb.set_style("darkgrid")
for (feat, tau), col in zip(top, sb.color_palette("Set1")[:nb]):
    sb.plt.plot(ranks, df[feat].tolist(), label=feat, c=col)
sb.plt.legend(loc="best")
sb.plt.xlabel('Diachrony', fontsize=10)
sb.plt.ylabel('Frequency', fontsize=10)
sb.plt.savefig('../figures/top_tau.pdf')
fig = sb.plt.figure()
sb.set_style("darkgrid")

for (feat, tau), col in zip(bottom, sb.color_palette("Set1")[:nb]):
    sb.plt.plot(ranks, df[feat].tolist(), label=feat, c=col)
sb.plt.legend(loc="best")
sb.plt.xlabel('Diachrony', fontsize=10)
sb.plt.ylabel('Frequency', fontsize=10)
sb.plt.savefig('../figures/bottom_tau.pdf')





