import glob
import os
import subprocess

filenames = [os.path.abspath(fn) for fn in glob.glob('../data/texts/*.txt')]
with open('spec_filenames.txt', 'w') as f:
    for fn in filenames:
        f.write(fn + '\n')

cmd = '''
java -cp "/Users/mike/GitRepos/panorama/src/stanford-corenlp/*" \
-Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
-annotators tokenize,ssplit,pos,lemma,ner \
-filelist spec_filenames.txt \
-outputFormat conll -outputDirectory ~/GitRepos/panorama/data/tagged/
'''

subprocess.call(cmd, shell=True)