#!/usr/bin/env python3
"""Fail-closed local delivery QA: links, actual bytes, range, ZIP and run-card.

No model fitting, registry download, live submission or slot selection here.
The stored complete audit is checked against its immutable candidate hash.
"""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit
import zipfile

ROOT=Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=set();self.downloads=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.add(attrs['id'])
        for key in ('href','src'):
            if attrs.get(key):self.links.append(attrs[key])
        if tag=='a' and 'download' in attrs:self.downloads.append(attrs.get('href',''))


def check_links(docs):
    errors=[];pages=list(docs.rglob('*.html'))
    for page in pages:
        parser=Links();parser.feed(page.read_text())
        for value in parser.links:
            url=urlsplit(value)
            if url.scheme or url.netloc or value.startswith(('mailto:','data:')):continue
            target=(page.parent/unquote(url.path)).resolve() if url.path else page
            if not target.is_relative_to(docs.resolve()):
                errors.append(f'{page}: local link escapes site: {value}');continue
            if target.is_dir():target=target/'index.html'
            if not target.is_file():errors.append(f'{page}: broken local link {value}');continue
            if url.fragment and target.suffix=='.html':
                other=Links();other.feed(target.read_text())
                if unquote(url.fragment) not in other.ids:errors.append(f'{page}: missing anchor {value}')
    return pages,errors


def check(root=ROOT):
    import sys
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import rasterio
    from gems59.gates import format_report
    docs=root/'docs'
    card=json.loads((root/'evidence/run_card_current.json').read_text())
    public=json.loads((docs/'data/run_card.json').read_text())
    assert card==public,'public card is stale'
    # --- Session-7 release contract -------------------------------------- #
    assert card['okay_to_submit'] is True and card['submission_slots_used']==0
    assert card['verdict']=='promote-candidate'
    assert 0<len(card['submission_note'])<=140
    assert card['submission_note_chars']==len(card['submission_note'])
    raster=root/card['file'];archive=root/card['zip_file']
    digest=hashlib.sha256(raster.read_bytes()).hexdigest()
    for source in (raster, archive):
        assert (root/'downloads'/source.name).read_bytes()==source.read_bytes(), 'legacy/root mirror is stale'
    assert digest==card['raster_sha256'],'TIFF changed after its audit'
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.namelist()==[raster.name],'ZIP must contain ONLY one TIFF'
        assert zipped.read(raster.name)==raster.read_bytes(),'ZIP contents differ'
    with rasterio.open(root/'data/official/sample_submission.tif') as ds:
        footprint=np.isfinite(ds.read(1))
    report=format_report(raster,root/'data/official/sample_submission.tif',footprint=footprint)
    assert report['ok'],report['problems']
    with rasterio.open(raster) as ds:
        assert ds.nodata is None
        values=ds.read(1)
    with rasterio.open(root/'data/official/labels.tif') as ds:
        known=ds.read(1)>0
    assert not (values[known]>0).any(),'positive predictions on known mask'
    assert int((values>0).sum())==card['emitted']
    assert card['validator_output']['checks'] and all(card['validator_output']['checks'].values())
    assert card['portal_preflight']['positive_pixels']==card['emitted']
    # registry clearance, all four legs
    audit=json.loads((root/'evidence/uniqueness_h57r.json').read_text())
    reg=card['correlation_overlap_vs_registry']
    assert audit['complete_accessible_scan']
    assert audit['registry_rasters_expected']==audit['registry_rasters_checked']
    assert audit['candidate_sha256']==digest
    assert not audit['identical_file_or_pixels']
    assert audit['leg3_scale_free']['passed'], 'scale-free duplicate test failed'
    assert audit['leg4_support_matched']['passed'], 'support-matched overlap test failed'
    assert reg['worst_spearman']<=0.90 and reg['worst_reverse_overlap']<=0.70
    assert reg['worst_jaccard']<=0.50 and reg['worst_dot_overlap_support_matched_peers']<=0.70
    assert reg['unique'] is True
    # the unrestricted one-sided leg is reported, not hidden
    assert 'leg1_literal_passed' in reg and reg['leg1_literal_passed'] is False
    assert reg['overlap_versus_peer_support']
    # the negative instrument result must still be on the record
    neg=json.loads((root/'evidence/offcat_instrument_check.json').read_text())
    assert neg['spearman_live_vs_proxy_full_density']['spearman_live_vs_proxy'] < -0.5
    port=json.loads((root/'evidence/portfolio_live_evidence.json').read_text())
    assert port['scored_rasters']>=15
    assert 'OWNER-REPORTED' in port['evidence_class']
    hyp=json.loads((root/'evidence/hypotheses_current.json').read_text())
    assert hyp['negative_results_this_session']
    sites=json.loads((root/'evidence/site_inventory.json').read_text())
    assert len(sites['repos'])==57
    pages,errors=check_links(docs)
    assert not errors,'\n'.join(errors)
    for name in ('index.html','executive-summary.html'):
        text=(docs/name).read_text();parser=Links();parser.feed(text)
        assert 'Submit to competition: OK' in text
        assert parser.downloads[0]==f'downloads/{raster.name}'
        assert parser.downloads[1]==f'downloads/{archive.name}'
        assert text.index('download-panel')<text.index('footer')
    js=(docs/'assets/site.js').read_text()
    assert 'localhost' not in js and '127.0.0.1' not in js
    for name in ('research.html','sources.html','results.html','irregularities.html'):
        assert 'ORGANIZER-CONFIRMED numbers as pasted' not in (docs/name).read_text()
    result=dict(pages_checked=len(pages),tiff_sha256=digest,tiff_bytes=raster.stat().st_size,
                zip_exactly_one_tiff=True,local_format_pass=True,links_pass=True,
                current_card_consistent=True,submission_cleared=True,
                registry_rasters_checked=audit['registry_rasters_checked'],
                worst_spearman=reg['worst_spearman'],
                worst_reverse_overlap=reg['worst_reverse_overlap'],
                worst_jaccard=reg['worst_jaccard'],
                worst_support_matched_overlap=reg['worst_dot_overlap_support_matched_peers'])
    print(json.dumps(result,indent=2))
    return result


def inspect_site(docs):
    """Compatibility helper for the repository's separate site-link smoke test."""
    pages, errors = check_links(Path(docs))
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',action='store_true')
    args=parser.parse_args()
    if args.build:
        from build_site import build
        build()
    check()


if __name__=='__main__':main()
