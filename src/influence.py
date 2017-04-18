import glob
import os
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sb
import numpy as np

MAX = 10000000000
cnt = 0

fns = sorted(list(glob.glob('../data/tagged/*.txt.conll')))
fns[::-1][:MAX]

estimates = {}

for filename in fns:
    id_ = os.path.basename(filename).replace('.conll', '')

    cnt += 1
    if cnt >= MAX:
        break 
    if cnt % 1000 == 0:
        print('-> doc', cnt)

    dates = []
    with open(filename, 'r') as f:
        doc = []
        for line in f.readlines():
            line = line.strip()
            if line:
                comps = line.split()
                tok, ner = comps[1], comps[4]
                if (len(tok) == 4 or len(tok) == 3) and tok.isdigit():
                    year = int(tok)
                    if year > 500 and year < 1600:
                        if len(tok) < 4:
                            dates.append('0' + tok)
                        else:
                            dates.append(tok)

    if len(dates) > 1:
        dates = Counter(d[:3] for d in dates)
        mf, f = dates.most_common(1)[0]
        if f > 1:
            if mf not in estimates:
                estimates[mf] = []
            estimates[mf].append(id_)

wiki_dir = '../data/wikified/'
ext = '.wikification.tagged.flat.html'
#target_categories={'person', 'writer'}

targets = set(l.strip() for l in open('golden_age_latin_writers.txt') if l.strip())

centuries = sorted(estimates)
docs = []
cnt = Counter()
for century in centuries:
    print(century)
    doc = []
    for id_ in estimates[century]:
        with open(wiki_dir + id_ + ext, 'r') as f:
            text = f.read()

        for node in BeautifulSoup(text, 'lxml').descendants:
            try:
                if node.name and node.name == 'a':
                    """
                    atts = set(node['cat'].lower().split())
                    if target_categories.intersection(atts) == target_categories:
                        id_ = '#' + node['href'].split('/')[-1]
                        doc.append(id_)
                    """
                    l = node['href'].split('/')[-1]
                    if l in targets:
                        id_ = '#' + node['href'].split('/')[-1]
                        doc.append(id_)

            except:
                continue

    cnt.update(doc)
    docs.append(doc)

def identity(x):
    return x

vectorizer = CountVectorizer(analyzer=identity)
X = vectorizer.fit_transform(docs).toarray()
row_sums = X.sum(axis=1)
X = X / row_sums[:, np.newaxis]
voc = vectorizer.get_feature_names()

fig = sb.plt.figure()
sb.set_style("darkgrid")

nb = 7
entities = [entity for entity, _ in cnt.most_common(nb)]
years = [int(c) for c in sorted(centuries)]

for entity, col in zip(entities, sb.color_palette("Set1")[:nb]):
    freqs = X[:, voc.index(entity)].tolist()
    sb.plt.plot(years, freqs, label=entity, c=col)


sb.plt.legend(loc='best')
sb.plt.xlabel('Diachrony (century)', fontsize=10)
sb.plt.ylabel('Frequency', fontsize=10)
sb.plt.savefig('../figures/entities_diachron.pdf')














    
    
