import os

from lxml import etree
from lxml.etree import fromstring

import PyPDF2

texts, dates = [], []


# for younger, xml files:
for dirpath, dirs, filenames in os.walk('../data/SPC_XML'):
    for filename in filenames:
        if not filename.endswith('.xml'):
            continue

        fn = os.sep.join((dirpath, filename))
        
        with open(os.path.abspath(fn)) as f:
            xml_str = f.read().encode('utf-8')
        try:
            root = etree.fromstring(xml_str, parser=etree.XMLParser(encoding='utf-8'))
            body = root.find('.//body')
            text = ''.join(body.itertext())
            year = root.find('.//year')
            dates.append(year.text)
            texts.append(text)

        except:
            print('-> parsing errors in:', fn)


# then older, ocr'ed files:
# for younger, xml files:
for dirpath, dirs, filenames in os.walk('../data/SPC_OCR'):
    
    for filename in filenames:
        if not filename.endswith('.pdf'):
            continue

        fn = os.path.abspath(os.sep.join((dirpath, filename)))

        xml_fn = fn.replace('a.pdf', 'h.xml')

        with open(os.path.abspath(xml_fn)) as f:
            xml_str = f.read().encode('utf-8')

        root = etree.fromstring(xml_str, parser=etree.XMLParser(encoding='utf-8'))
        year = root.find('.//copyright-year')
            
        text = ''
        with open(fn, 'rb') as pdfFileObj:
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            for p in range(pdfReader.numPages):
                pageObj = pdfReader.getPage(p)
                text += ' ' + pageObj.extractText()
        
        texts.append(text)
        dates.append(year.text)

try:
    shutil.rmtree('../data/texts')
except:
    pass

try:
    os.mkdir('../data/texts')
except:
    pass


print(len(texts))
print(len(dates))

cnt = 0
for t, d in zip(texts, dates):

    cnt += 1
    with open('../data/texts/' + d + '_' + str(cnt) + '.txt', 'w') as f:
        f.write(t)






