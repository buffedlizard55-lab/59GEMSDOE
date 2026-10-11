#!/usr/bin/env python3
"""Bounded content review of each homepage at the inventoried immutable commit.

Retrieves public HTML through GitHub, not a scoring receipt. Records title,
heading and linked-TIFF evidence without treating a site's score claims as
organizer-confirmed. Cached source bytes are ignored, the summary is durable.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OWNER='buffedlizard55-lab'


class Review(HTMLParser):
    def __init__(self):
        super().__init__();self.title=[];self.heading=[];self.links=[];self.mode=None
    def handle_starttag(self,tag,attrs):
        if tag=='title':self.mode='title'
        if tag in ('h1','h2'):self.mode='heading'
        if tag=='a':
            href=dict(attrs).get('href','')
            if '.tif' in href.lower() or '.zip' in href.lower():self.links.append(href)
    def handle_endtag(self,tag):
        if tag in ('title','h1','h2'):self.mode=None
    def handle_data(self,data):
        if self.mode=='title':self.title.append(data.strip())
        if self.mode=='heading':self.heading.append(data.strip())


def review(item):
    base=dict(repo=item['repo'],commit=item['commit'],website=item['website'],
              review_scope='Pinned homepage content/link review, NOT code reproduction or organizer score confirmation')
    indexes=item['index_files']
    if not indexes:return dict(base,status='no index found in pinned tree')
    path='docs/index.html' if 'docs/index.html' in indexes else indexes[0]
    endpoint=f'repos/{OWNER}/{item["repo"]}/contents/{path}?ref={item["commit"]}'
    try:
        response=subprocess.run(['gh','api',endpoint,'-H','Accept: application/vnd.github.raw+json'],check=True,capture_output=True,timeout=60)
        raw=response.stdout
        cache=ROOT/'.cache/site_reviews';cache.mkdir(parents=True,exist_ok=True)
        (cache/f'{item["repo"]}.html').write_bytes(raw)
        text=raw.decode('utf-8');parser=Review();parser.feed(text)
        return dict(base,status='retrieved',path=path,source_url=f'https://github.com/{OWNER}/{item["repo"]}/blob/{item["commit"]}/{path}',
                    html_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                    title=' '.join(parser.title),headings=parser.heading,
                    linked_rasters_or_zips=sorted(set(parser.links)),
                    receipt_verified=False)
    except Exception as error:
        return dict(base,status='error',error=str(error)[:200])


def main():
    inventory=json.loads((ROOT/'evidence/site_inventory.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(review,inventory['repos']))
    result=dict(evidence_class='THIRD-PARTY PAGE REVIEW, NOT organizer score confirmation',
                reviewed_utc=datetime.now(timezone.utc).isoformat(),
                pages_retrieved=sum(r['status']=='retrieved' for r in rows),rows=rows)
    (ROOT/'evidence/sibling_page_reviews.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('reviewed_utc','pages_retrieved')},indent=2))
    return 1 if any(r['status']=='error' for r in rows) else 0


if __name__=='__main__':raise SystemExit(main())
