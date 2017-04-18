import glob
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()


year_counter = {}
for filename in sorted(glob.glob('../data/tagged/*.txt.conll')):
    year = int(os.path.basename(filename).split('_')[0])
    if year not in year_counter:
        year_counter[year] = 0

    with open(filename, 'r') as f:
        y = os.path.basename(filename).split('_')[0]
        for line in f.readlines():
            line = line.strip()
            if line:
                year_counter[year] += 1

year_counts = sorted(year_counter.items(), key=itemgetter(0), reverse=True)
years, cnts = zip(*year_counts)
years = [int(y) for y in years]

nb_words = sum(cnts)

sb.plt.barh(years, cnts, color='lightslategray')
sb.plt.title('Speculum archive (' + str(min(years)) + \
                '-' +  str(max(years)) + ')\n'\
                +str(nb_words)+' tokens  in total')
sb.plt.xlabel('Number of words')
sb.plt.ylabel('Year')
sb.plt.savefig('../figures/nb_words.pdf')
