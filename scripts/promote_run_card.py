#!/usr/bin/env python3
"""Promote the Session-7 build receipts into ``evidence/run_card_current.json``.

The site, the README and the release guard all read one card.  This script
composes it from receipts that were written by other scripts; it computes
nothing scientific of its own and refuses to promote if any clearance is missing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gems59.uniqueness import JACCARD_LIMIT, OVERLAP_LIMIT, RHO_LIMIT   # noqa: E402


def load(name):
    return json.loads((ROOT / 'evidence' / f'{name}.json').read_text())


def main() -> int:
    build = load('run_card_h57r')
    construction = load('h57r_build')
    uniq = load('uniqueness_h57r')
    port = load('portfolio_live_evidence')
    instr = load('offcat_instrument_check')
    struct = load('offcat_structure')
    hyp = load('hypotheses_current')

    leg1 = uniq['leg1_literal']
    leg2 = uniq['leg2_representation_aware']
    leg3 = uniq['leg3_scale_free']
    leg4 = uniq['leg4_support_matched']
    worst_rho = max((r['spearman_full_footprint'] or 0.0) for r in uniq['rows']) \
        if uniq['rows'] else None
    # The literal inherited gate is operative. Representation-aware, reverse-
    # overlap, Jaccard, and support-matched summaries are diagnostics only and
    # cannot override a literal rank/forward-overlap firing.
    unique = bool(uniq['complete_accessible_scan'] and leg1['passed']
                  and not uniq['identical_file_or_pixels'])
    if not uniq['complete_accessible_scan']:
        raise SystemExit('refusing to promote: the registry scan is incomplete')

    angle = {f"{r['bin_lo_deg']:.0f}-{r['bin_hi_deg']:.0f}": r
             for r in port['relative_strike_correlations']}
    card = dict(
        evidence_class='SUBMISSION-RELEASE (composition of receipts; not a score)',
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        session=7,
        hypothesis=hyp['hypotheses'][0]['title'],
        mechanism=('Displacement-weighted damage-zone intensity with a kernel width that grows '
                   'with host system length; the axial orientation of that intensity field is '
                   'the strand direction; the emitted dots are apportioned across joint '
                   '(distance, relative-strike) strata whose target is the measured distribution '
                   'of genuinely new faults, with the sub-parallel class down-weighted by a '
                   'single pre-declared factor.'),
        named_non_fault_process_that_could_mimic_it=(
            'Range-front erosional scarps, fluvial terrace margins and lithologic contacts '
            'that sit between unrelated fault systems, plus survey flight-line levelling '
            'residuals in the aeromagnetic grid.'),
        submission_name=build['submission_name'],
        submission_note=build['submission_note'],
        submission_note_chars=build['note_characters'],
        file=build['file'], zip_file=build['zip_file'],
        raster_sha256=build['sha256'], bytes=build['bytes'],
        validator_output=build['validator_output'],
        portal_preflight=build['portal_preflight'],
        catalogue_clearance=build['catalogue_clearance'],
        construction=construction['parameters'],
        emitted=construction['emitted_pixels'],
        emitted_distance_quantiles_px=construction['emitted_distance_quantiles'],
        emitted_angle_histogram=construction['emitted_angle_hist'],
        target_weights=construction['target_weights'],
        proxy_reference=construction['proxy_reference'],
        correlation_overlap_vs_registry=dict(
            registry_rasters_expected=uniq['registry_rasters_expected'],
            registry_rasters_checked=uniq['registry_rasters_checked'],
            complete_accessible_scan=uniq['complete_accessible_scan'],
            worst_spearman=worst_rho,
            worst_spearman_submission=next(
                (r['submission'] for r in uniq['rows']
                 if r['spearman_full_footprint'] == worst_rho), None),
            worst_dot_overlap_dot_peers=leg4['worst_forward_overlap'],
            worst_dot_overlap_support_matched_peers=leg4['worst_forward_overlap'],
            support_matched_peers=leg4['peers'],
            support_matched_band=[leg4['band_lo'], leg4['band_hi']],
            worst_dot_overlap_any_peer=leg1['worst_dot_overlap'],
            worst_reverse_overlap=leg3['worst_reverse_overlap'],
            worst_jaccard=leg3['worst_jaccard'],
            identical_file_or_pixels=uniq['identical_file_or_pixels'],
            rho_limit=RHO_LIMIT, overlap_limit=OVERLAP_LIMIT,
            jaccard_limit=JACCARD_LIMIT,
            representation_classes=dict(dot=leg2['dot_peers'], dense=leg2['dense_peers']),
            spearman_subgrid_fraction=uniq['spearman_subgrid_fraction'],
            leg1_literal_passed=leg1['passed'],
            leg2_representation_aware_passed=leg2['passed'],
            leg3_scale_free_passed=leg3['passed'],
            leg4_support_matched_passed=leg4['passed'],
            overlap_versus_peer_support=uniq['overlap_versus_peer_support'],
            protocol_note=('The literal thresholds apply to every indexed raster. Any rho > 0.90 '
                           'or forward 3-px overlap > 0.70 is a STOP regardless of peer density or '
                           'representation. Reverse overlap, Jaccard, support-matched, and '
                           'representation-aware summaries are diagnostic only and cannot clear '
                           'a literal firing.'),
            representation_aware_diagnostic_only=True,
            unique=unique),
        portfolio_live_evidence=dict(
            evidence_class=port['evidence_class'],
            scored_rasters=port['scored_rasters'],
            caveat=port['caveats'],
            label_provenance=port['label_provenance'],
            frac_within_2px_of_catalogue=port['property_correlations']['frac_within_2px_of_catalogue'],
            log_n_dots=port['property_correlations']['log_n_dots'],
            distance_p50_px=port['property_correlations']['distance_p50_px'],
            angle_0_5_deg=angle.get('0-5'), angle_25_30_deg=angle.get('25-30'),
            angle_35_40_deg=angle.get('35-40')),
        instrument_negative_result=dict(
            evidence_class=instr['evidence_class'],
            spearman_live_vs_proxy_full_density=instr['spearman_live_vs_proxy_full_density'],
            conclusion=instr['conclusion']),
        holdout_dti=dict(
            evidence_class='NOT APPLICABLE - the inherited catalogue holdout was NOT used to '
                           'select this placement; see instrument_negative_result and '
                           'portfolio_live_evidence',
            withheld_positive_pixels=int(struct.get('proxy_pixels_total', 0)),
            proxy_structure_measurement='evidence/offcat_structure.json'),
        holdout_dot_dti=dict(
            evidence_class='NOT APPLICABLE - no holdout DTI is claimed for this raster; the '
                           'off-catalogue proxy instrument was measured and rejected as a '
                           'ranking instrument before any placement was built'),
        irregularities=['IR-57-INSTR-01', 'IR-57-SGMC-01', 'IR-57-REPR-01'],
        verdict='promote-candidate' if unique else 'negative',
        okay_to_submit=bool(unique),
        submission_slots_used=0,
        generated_utc=build['generated_utc'])
    (ROOT / 'evidence/run_card_current.json').write_text(
        json.dumps(card, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: card[k] for k in
                      ('verdict', 'okay_to_submit', 'emitted', 'raster_sha256',
                       'submission_note_chars')}, indent=2))
    print(json.dumps(card['correlation_overlap_vs_registry'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())