#!/usr/bin/env python
"""Fetch Spider-Man 3 script from scripts.com"""
import sys, re, time
sys.stdout.reconfigure(encoding='utf-8')
import requests
from bs4 import BeautifulSoup

all_text = []
page = 1
while True:
    if page > 1:
        url = f'https://www.scripts.com/script/spider-man_3_18657/{page}'
    else:
        url = 'https://www.scripts.com/script/spider-man_3_18657'
    r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
    if r.status_code != 200:
        print(f'Page {page}: status {r.status_code}, stopping')
        break
    soup = BeautifulSoup(r.text, 'html.parser')
    script_div = soup.find('div', class_='fullscript')
    if not script_div:
        for div in soup.find_all('div'):
            text = div.get_text()
            if len(text) > 2000 and 'PETER' in text:
                script_div = div
                break
    if script_div:
        text = script_div.get_text()
        all_text.append(text)
        print(f'Page {page}: {len(text):,} chars')
    else:
        print(f'Page {page}: no text found, stopping')
        break
    next_link = soup.find('a', string=re.compile(r'Next|next|>>'))
    if not next_link:
        print(f'No next link at page {page}, stopping')
        break
    page += 1
    if page > 30:
        break
    time.sleep(0.5)

full_text = '\n'.join(all_text)
print(f'\nTotal: {len(full_text):,} chars from {page} pages')
print(f'First 300 chars: {full_text[:300]}')
with open('/tmp/spiderman3.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)
