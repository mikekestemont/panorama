from bs4 import BeautifulSoup
import urllib.request

resp = urllib.request.urlopen("https://en.m.wikipedia.org/wiki/Category:Golden_Age_Latin_writers")
soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))

f = open('golden_age_latin_writers.txt', 'w')
for link in soup.find_all('a', href=True):
    l = link['href']
    if l.startswith('/wiki/') and ':' not in l:
        f.write(l.replace('/wiki/', '')+'\n')
f.close()