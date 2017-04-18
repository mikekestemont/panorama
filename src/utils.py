import pickle
from collections import Counter

def parse_wikified_files(wiki_dir='../data/wikified/',
                ext='.wikification.tagged.flat.html',
                max_nb_files=None,
                outputf='../data/wikified.txt'):

    link_dict = dict()
    bf = open(outputf, 'w')

    fns = list(glob.glob(wiki_dir + '*' + ext))
    fns = sorted(fns)

    if max_nb_files:
        fns = fns[:max_nb_files]

    for i, fn in enumerate(fns):
        if i and i % 500 == 0:
            print('- yielded', i, 'files')

        with open(fn, 'r') as f:
            tokens, meta = parse_wikified_file(f.read())

            for item in meta:
                if item not in link_dict:
                    link_dict[item] = meta[item]

            bf.write(' '.join(tokens) + '\n')

    pickle.dump(link_dict, open('../data/link_dict.p', 'wb'))
    bf.close()

def parse_wikified_file(text):
    tokens = []
    flagged = False
    meta = dict()
    for node in BeautifulSoup(text, 'lxml').descendants:
        try:
            if not node.name:
                if flagged:
                    flagged = False
                    continue
                else:
                    toks = [c.lower() for c in node.split() \
                                if c not in punct]
                    if toks:
                        tokens.extend(toks)
            elif node.name == 'a':
                id_ = '#' + node['href'].split('/')[-1]
                atts = set(node['cat'].lower().split())
                if atts:
                    meta[id_] = atts
                tokens.append(id_)
                flagged = True
        except KeyError:
            pass

    return tuple(tokens), meta


class WikifiedSpeculumSentences(object):

    def __init__(self, wikifile='../data/wikified.txt',
                max_nb_files=None,
                target_categories=None):

        self.target_categories = target_categories
        self.max_nb_files = max_nb_files
        self.wikifile = wikifile
        self.counter = Counter()
        self.link_dict = pickle.load(open('../data/link_dict.p', 'rb'))
 
    def __iter__(self):
        cnt = 0
        for line in open(self.wikifile, 'r'):
            cnt += 1
            if cnt >= self.max_nb_files:
                break
            if cnt and cnt % 500 == 0:
                print('- yielded', cnt, 'files')

            tokens = line.strip().split()

            for token in tokens:
                try:
                    atts = self.link_dict[token]
                    if self.target_categories:
                        if atts and self.target_categories.intersection(atts) == self.target_categories:
                            self.counter[token] += 1
                    else:
                        self.counter[token] += 1

                except KeyError:
                    continue
                    
            yield tokens