import glob

import nltk.data
from nltk.tokenize import TreebankWordTokenizer

import gensim

tokenizer = TreebankWordTokenizer()

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []
for filename in glob.glob('../data/texts/*.txt'):
    with open(filename, 'r') as f:
        text = f.read()
        curr_sents = sent_detector.tokenize(text.strip())
        curr_sents = [tokenizer.tokenize(s.strip()) for s in curr_sents if s.strip()]
        sentences.extend(curr_sents)

print('running bigram transformer')
bigram_transformer = gensim.models.Phrases(sentences)
sentences = bigram_transformer[sentences]

print('running word2vec model')
model = gensim.models.Word2Vec(sentences, min_count=10)

model.init_sims(replace=True)

print(model.similar_by_word('manuscript', topn=10))
print(model.similar_by_word('Clairvaux', topn=10))
print(model.similar_by_word('Rome', topn=10))