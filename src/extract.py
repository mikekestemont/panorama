import os
import re
import subprocess
import codecs

from lxml import etree
from lxml.etree import fromstring

linebreak = re.compile(r'\-\s*\n\s*')


def main():

    try:
        shutil.rmtree('../data/texts')
    except:
        pass

    try:
        os.mkdir('../data/texts')
    except:
        pass

    cnt = 0

    # for younger, xml files:
    for dirpath, dirs, filenames in os.walk('../data/SPC_XML'):
        for filename in filenames:
            if not filename.endswith('.xml'):
                continue

            fn = os.sep.join((dirpath, filename))
            
            with codecs.open(os.path.abspath(fn), 'r', 'utf-8') as f:
                xml_str = f.read().encode('utf-8')
            
            try:                
                root = etree.fromstring(xml_str, parser=etree.XMLParser(encoding='utf-8'))
                body = root.find('.//body')
            except:
                print('-> parsing errors in:', fn)
                continue

            if body is not None:
                text = ''.join(body.itertext())
                year = root.find('.//year').text
                
                cnt += 1
                with codecs.open('../data/texts/' + year + '_' + str(cnt) + '.txt', 'w', 'utf-8') as f:
                    f.write(text)

    # then older, ocr'ed files:
    # for younger, xml files:
    for dirpath, dirs, filenames in os.walk('../data/SPC_PDF'):
        
        for filename in filenames:
            if not filename.endswith('.pdf'):
                continue

            print(filename)
            fn = os.path.abspath(os.sep.join((dirpath, filename)))

            xml_fn = fn.replace('a.pdf', 'h.xml')

            with open(os.path.abspath(xml_fn)) as f:
                xml_str = f.read().encode('utf-8')

            root = etree.fromstring(xml_str, parser=etree.XMLParser(encoding='utf-8'))
            year = root.find('.//copyright-year').text
            
            subprocess.call('java -jar pdfbox-app-2.0.3.jar ExtractText '+fn+' tmp.txt',
                shell = True)

            with open('tmp.txt', 'r') as f:
                text = f.read()
                text = re.sub(linebreak, '', text)

            cnt += 1
            with codecs.open('../data/texts/' + year + '_' + str(cnt) + '.txt', 'w', 'utf-8') as f:
                    f.write(text)

    os.remove('tmp.txt')

    print('-> extracted', cnt, 'items')


if __name__ == '__main__':
    main()





