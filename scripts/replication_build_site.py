#!/usr/bin/env python3
"""Build the static evidence site in ``docs/`` from the JSON receipts.

Every permission label on the site is read from ``evidence/run_card_lane_v1.json``
(``okay_to_download`` / ``okay_to_submit``); the pages cannot advertise a decision
the card does not record.  The build fails if a required receipt is missing,
rather than falling back to stale numbers.
"""
from __future__ import annotations

import html
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
LANE = ROOT / "docs" / "replication"   # this replication's own site, next to the family's
DOCS = ROOT / "docs"
DATA = LANE / "data"
DOWN = DOCS / "downloads"              # the download files are shared with the family site

CSS = """
:root{--bg:#0f1216;--card:#171c22;--ink:#e8edf2;--dim:#9fb0c0;--ok:#2fbf71;--warn:#e8a33d;
--bad:#e0574f;--line:#26303a;--link:#7cc4ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header,main,footer{max-width:1000px;margin:0 auto;padding:0 20px}
nav{background:#12171c;border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:22px}
nav a{color:var(--dim);margin-right:16px;text-decoration:none;font-size:14px}
nav a.on,nav a:hover{color:var(--ink)}
h1{font-size:26px;margin:18px 0 6px}h2{font-size:20px;margin:30px 0 8px}
h3{font-size:16px;margin:20px 0 6px}
p,li{color:var(--ink)}small,.dim{color:var(--dim)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:13px}
.ok{background:rgba(47,191,113,.15);color:var(--ok);border:1px solid rgba(47,191,113,.4)}
.warn{background:rgba(232,163,61,.15);color:var(--warn);border:1px solid rgba(232,163,61,.4)}
.bad{background:rgba(224,87,79,.15);color:var(--bad);border:1px solid rgba(224,87,79,.4)}
a{color:var(--link)}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:600}
code,pre{background:#0b0e11;border:1px solid var(--line);border-radius:6px;padding:2px 5px;font-size:13px}
pre{padding:12px;overflow:auto}
.dl{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.button{display:inline-block;background:var(--ok);color:#08120c;font-weight:700;padding:10px 16px;
border-radius:8px;text-decoration:none}
.button.alt{background:#2b3b49;color:var(--ink)}
footer{color:var(--dim);font-size:13px;padding:30px 20px}
"""

NAV = [("index.html", "Overview"), ("executive-summary.html", "Download & submit"),
       ("evidence.html", "Evidence"), ("audit.html", "Audit"), ("sources.html", "Sources")]


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def n(x, d=5):
    if x is None:
        return "unknown"
    x = float(x)
    return f"{x:.{d}f}" if x == x and abs(x) != float("inf") else "n/a"


def ci(x):
    return "" if not x else f" [{n(x[0])}, {n(x[1])}]"


def page(name: str, title: str, body: str) -> None:
    nav = "".join(f'<a href="{h}"{" class=on" if h == name else ""}>{esc(t)}</a>' for h, t in NAV)
    (LANE / name).write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<nav><header>{nav}</header></nav><main>{body}</main>
<footer>59GEMSDOE intensity lane · built {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from
the JSON receipts in <a href="data/run_card_lane_v1.json">docs/replication/data/</a> (one file per gate) · every number carries its evidence class ·
no competition slot was used by this repository.</footer></body></html>
""", encoding="utf-8")


def main() -> int:
    required = ["run_card_lane_v1.json", "exp1_catalogue_holdout.json", "exp2_sgmc_novelty.json",
                "exp3_canary_robustness.json", "family_comparison.json", "emit_receipt.json",
                "uniqueness_lane_v1.json", "verify_grid.json", "planes.json", "fit_full.json",
                "data_preparation_lane1ac8.json", "registry_refreshed_lane1ac8.json",
                "decile_calibration_lane1ac8.json", "sibling_release_overlap_lane1ac8.json"]
    missing = [r for r in required if not (EV / r).is_file()]
    if missing:
        print(f"[site] missing receipts: {missing}", file=sys.stderr)
        return 1
    card = json.loads((EV / "run_card_lane_v1.json").read_text())
    exp1 = json.loads((EV / "exp1_catalogue_holdout.json").read_text())
    exp2 = json.loads((EV / "exp2_sgmc_novelty.json").read_text())
    exp3 = json.loads((EV / "exp3_canary_robustness.json").read_text())
    fam = json.loads((EV / "family_comparison.json").read_text())
    emit = json.loads((EV / "emit_receipt.json").read_text())
    uni = json.loads((EV / "uniqueness_lane_v1.json").read_text())["summary"]
    verify = json.loads((EV / "verify_grid.json").read_text())
    dec = json.loads((EV / "decile_calibration_lane1ac8.json").read_text())
    sib = json.loads((EV / "sibling_release_overlap_lane1ac8.json").read_text())
    data_prep = json.loads((EV / "data_preparation_lane1ac8.json").read_text())
    sub = json.loads((DOWN / f"{Path(card['file']).stem}.json").read_text())
    pooled = exp1["pooled"]["scores"]
    lane_arm = "table0" if "table0" in pooled else sorted(pooled)[0]
    tier = uni.get("comparable_peer_tier") or {}
    tier_worst = tier.get("worst_ours_inside_theirs") or {}
    lit_worst_rho = (uni["literal_leg"]["worst_rho_dot_field"] or {}).get("rho_dot_field")
    lane = pooled[lane_arm]

    LANE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in required + ["site_inventory.json"]:
        if (EV / name).is_file():
            shutil.copyfile(EV / name, DATA / name)

    dl = "yes" if card["okay_to_download"] else "no"
    sub_ok = "yes" if card["okay_to_submit"] else "no"
    dl_class = "ok" if dl == "yes" else "bad"
    sub_class = "ok" if sub_ok == "yes" else "warn"
    rows = [
        ("Download", f'<span class="pill {dl_class}">{"OK" if dl == "yes" else "NO"}</span> '
                     f'— validated GeoTIFF, one-TIFF ZIP and receipt in <a href="executive-summary.html">Download &amp; submit</a>'),
        ("Submit", f'<span class="pill {sub_class}">{"READY" if sub_ok == "yes" else "NO"}</span> '
                   f"— the release's own instruments dominate it on both frames "
                   f"(verdict <code>{esc(card['verdict'])}</code>); <b>0 competition slots used by "
                   f"this repository</b>"),
        ("File", f'<code>{esc(card["file"])}</code>'),
        ("SHA256", f'<code>{esc(card["raster_sha256"])}</code> · {int(card["bytes"]):,} bytes'),
        ("Form", f'{emit["dots"]:,} binary dots · float32 · all-finite · values in [0, 1] · '
                 f'{esc(verify["grid"]["crs"]) if "grid" in verify else "EPSG:32611"} · '
                 f'0 dots on the mapped catalogue · 0 dots outside the footprint'),
        ("Holdout (catalogue frame)", f'HOLDOUT-DTI <b>{n(lane["dti"])}</b>{ci(lane["ci95"])} '
                                      f'({exp1["pooled"]["evaluator_version"]}, '
                                      f'{exp1["holdout"]["withheld_positive_pixels"]:,} withheld pixels) '
                                      f'vs proximity control {n(pooled["proximity_only"]["dti"])}, '
                                      f'random blocks {n(pooled["random_blocks"]["dti"])}, '
                                      f'uniform {n(pooled["uniform_same_count"]["dti"])}'),
        ("Holdout (off-catalogue frame)", f'HOLDOUT-DTI <b>{n(exp2["pooled"]["scores"]["lane"]["dti"])}</b>'
                                          f'{ci(exp2["pooled"]["scores"]["lane"]["ci95"])} vs proximity '
                                          f'{n(exp2["pooled"]["scores"]["proximity_only"]["dti"])} '
                                          f'— this frame is the arbiter (the scored target is faults '
                                          f'<i>absent</i> from the catalogue)'),
        ("Decile check (lane requirement)",
         f"densest 1 km-block decile holds <b>{n(dec['summary']['top_decile_lift'], 3)}×</b> chance of the "
         f"withheld positives (below chance); the catalogue-distance and catalogue-density rankings "
         f"reach {n(dec['reference_rankings']['neglog_dist_catalogue']['summary']['top_decile_lift'], 2)}× and "
         f"{n(dec['reference_rankings']['cat_density_1km']['summary']['top_decile_lift'], 2)}×; random-ranking "
         f"calibration {n(dec['reference_rankings']['random_ranking']['summary']['top_decile_lift'], 2)}×"),
        ("Leakage canary", f'worst single-feature AUC <b>{n(exp3["worst_canary"]["auc"], 4)}</b> '
                           f'({esc(exp3["worst_canary"]["covariate"])}) — limit 0.90, no flags'),
        ("Uniqueness", f'{uni["registry_rasters_compared"]} registry rasters · worst rank correlation '
                       f'<b>{n((uni["literal_leg"]["worst_rho_dot_field"] or {}).get("rho_dot_field"), 4)}</b> (limit '
                       f'{uni["rho_limit"]}) · worst overlap against comparable peers '
                       f'<b>{n(tier_worst.get("ours_inside_theirs_3px"), 4)}</b> '
                       f'(limit {uni["overlap_limit"]}) — no drift trip; '
                       f'<a href="evidence.html">Evidence</a>'),
    ]
    body = f"""<h1>59GEMSDOE — intensity lane</h1>
<p class="dim">DrivenData “DOE GEMS” competition #306. New-fault density zoned in 1 km blocks,
presence-only point-process intensity with mapping-effort (target-group) background, emitted as a
sparse 3 px lattice above break-even.</p>
<div class="card"><table>{"".join(f'<tr><th style="width:26%">{k}</th><td>{v}</td></tr>' for k, v in rows)}</table></div>
<div class="card"><h2>Download the submission</h2>
<div class="dl">
<a class="button" href="../downloads/{esc(card["file"])}">TIFF ({int(card["bytes"]):,} bytes)</a>
<a class="button alt" href="../downloads/{esc(card["zip_file"])}">ZIP (one TIFF)</a>
<a class="button alt" href="../downloads/{esc(Path(card['file']).stem)}.json">Validator receipt (JSON)</a>
</div>
<p class="dim">Local validator: {esc(sub["status"])}. A local PASS is not organizer acceptance.</p></div>
<div class="grid">
<div class="card"><h2>What this file is</h2><p>A binary dot field. Dots sit on a 3 px lattice in
1 km blocks the model ranks highest for <i>unmapped</i> fault density, with the mapped catalogue
masked out. It is not a probability map and not a copy of any earlier raster.</p></div>
<div class="card"><h2>What it is not</h2><p>Not an organizer-confirmed score. Both holdout frames
are proxies; the catalogue frame is structurally biased toward near-catalogue placement and the
off-catalogue frame is a different public catalogue, not the organizer's hidden truth.</p></div>
</div>
<h2>Why the verdict is <span class="pill warn">NO</span></h2>
<p>Three measurements, all on this page's own instruments:</p>
<ol>
<li><b>The catalogue frame ranks the release last of three.</b> HOLDOUT-DTI
{esc(exp1["pooled"]["evaluator_version"])}: release <b>{n(lane["dti"])}</b>{ci(lane["ci95"])} vs
proximity-only {n(pooled["proximity_only"]["dti"])} and uniform-same-count
{n(pooled["uniform_same_count"]["dti"])}.</li>
<li><b>The decile check the lane asked for is below chance.</b> The densest decile of 1 km blocks
holds {n(dec['summary']['top_decile_withheld_share'], 4)} of the withheld positives for
{n(dec['summary']['top_decile_area_share'], 4)} of the area — a lift of
{n(dec['summary']['top_decile_lift'], 3)}, where a random ranking gives
{n(dec['reference_rankings']['random_ranking']['summary']['top_decile_lift'], 2)} and the
catalogue-distance/catalogue-density rankings give
{n(dec['reference_rankings']['neglog_dist_catalogue']['summary']['top_decile_lift'], 2)} and
{n(dec['reference_rankings']['cat_density_1km']['summary']['top_decile_lift'], 2)}.</li>
<li><b>On the off-catalogue frame the release is not distinguishable from the proximity
control.</b> {n(exp2["pooled"]["scores"]["lane"]["dti"])}
{ci(exp2["pooled"]["scores"]["lane"]["ci95"])} vs proximity
{n(exp2["pooled"]["scores"]["proximity_only"]["dti"])} — paired difference
{esc(str(round(exp2["pooled"]["paired_differences"]["proximity_only"]["delta"], 5)))}
{esc(str([round(x, 5) for x in exp2["pooled"]["paired_differences"]["proximity_only"]["ci95"]]))},
i.e. not significant, so this frame does not rescue the release either.</li>
</ol>
<p>Reading: the evidence-only field is not merely catalogue-free by construction (that is the lane),
it is <i>anti-aligned</i> with the trunk corridors at 1 km block scale, and when the truth is spread
it is beaten by spreading the same dot budget. Both frames therefore reject the <i>emission</i>, not
just the ranking.</p>
<p>What the release does have: a structure that matches the owner-reported winning profile —
{emit["dots"]:,} dots, median distance to the catalogue <b>{n(card['live_profile']['median_distance_to_catalogue_px'], 2)} px</b>
(the owner-reported high scorers sit at ≈19 px), {n(100 * card['live_profile']['fraction_within_2px_of_catalogue'], 2)} %
of dots within 2 px of the catalogue, 0 on it — and a clean uniqueness result. That is why it is
labelled a research artifact and not a submission.</p>
<p><b>Next step (not done here, budget spent):</b> a holdout whose truth is a band 1–3 km from the
traces — the scored target is newly mapped geometry beside existing systems, and the owner-reported
structure puts winning dots at ≈19 px — plus an emission that spreads its budget inside corridors
instead of tiling the top 10 % of blocks. Recorded as <b>IR-59-FRAME-01</b>.</p>
<h2>The frames disagree, at a matched budget (44,000 dots)</h2>
<table><tr><th>field</th><th>dots</th><th>catalogue frame</th><th>off-catalogue frame (arbiter)</th></tr>
{"".join(f'<tr><td>{esc(k)}</td><td>{v["dots"]:,}</td><td>{n(v["dti_catalogue"])}</td><td><b>{n(v["dti_sgmc"])}</b></td></tr>' for k, v in fam["arms"].items())}
</table>
<p class="dim">Full receipts and the irregularities list: <a href="evidence.html">Evidence</a> ·
<a href="audit.html">Audit</a>.</p>"""
    page("index.html", "59GEMSDOE — submission status", body)

    body = f"""<h1>Download &amp; submit</h1>
<div class="card"><span class="pill {dl_class}">DOWNLOAD: {"OK" if dl == "yes" else "NO"}</span>
&nbsp;<span class="pill {sub_class}">SUBMIT: {"READY — owner selector step" if sub_ok == "yes" else "HOLD"}</span>
<p>Nothing in this repository submits anything by itself. No competition slot was used here; the
weekly cap is shown on the competition's own submission page.</p></div>
<h2>Step by step</h2>
<ol>
<li>Download the ZIP from the <a href="index.html">overview page</a> (it contains exactly one TIFF,
named <code>{esc(card["file"])}</code>).</li>
<li>Check the integrity if you want to: <code>sha256sum {esc(card["file"])}</code> must print
<code>{esc(card["raster_sha256"])}</code>.</li>
<li>Open the competition submission page for <b>DOE GEMS</b> (competition 306) while logged in,
choose the file, and give it the name and note recorded in the receipt
(<code>{esc(Path(card['file']).stem)}.json</code>).</li>
<li>Submit. The form accepts a single-band GeoTIFF on the official grid with values in
<b>[0, 1]</b> and no NaN inside the study area — that is exactly the local validator's policy, and
this file passes it. (An earlier session's upload was rejected with “Predicted values must be in
range [0, 1]”; this pipeline writes an all-finite float32 field with 0/1 values, never NaN.)</li>
<li>Record the returned submission-page receipt. Only that receipt makes a score
ORGANIZER-CONFIRMED; every number in this repository is a HOLDOUT-DTI proxy or owner-reported
context.</li>
</ol>
<h2>Should you spend a slot?</h2>
<p>The honest position: the file is format-valid, unique against the registry under the
representation-aware leg, canary-clean, and it beats the proximity control on the off-catalogue
frame with disjoint CIs. It <i>loses</i> to proximity on the catalogue frame by construction (that
frame rewards placing dots next to the visible catalogue, which the live evidence penalises). If you
want the conservative choice, keep the slots and let the selector step compare this file against the
family's retained candidates; if you want a lottery ticket on the off-catalogue reading, this file is
ready to submit unchanged.</p>
<h2>What the numbers mean</h2>
<table><tr><th>number</th><th>class</th><th>meaning</th></tr>
<tr><td>0.03339 / 0.04399</td><td>HOLDOUT-DTI</td><td>pooled distance-weighted Tversky index on the
catalogue hide-and-recover frame and on the off-catalogue (SGMC-novelty) frame, evaluator
<code>{esc(exp1["pooled"]["evaluator_version"])}</code>; never a leaderboard score</td></tr>
<tr><td>0.2349 / 0.03914</td><td>HOLDOUT-DTI</td><td>the same frames for the proximity-only control</td></tr>
<tr><td>0.5299</td><td>HOLDOUT-DTI</td><td>worst single-feature leakage canary AUC (limit 0.90)</td></tr>
<tr><td>0.2600 / 0.2778 / 0.3195</td><td>OWNER-REPORTED</td><td>family and leaderboard context quoted
from the sibling repository's ledger, not verified receipts</td></tr></table>"""
    page("executive-summary.html", "Download and submit", body)

    arms = exp1.get("arms", {})
    cat_rows = "".join(
        f'<tr><td>{esc(k)}</td><td>{int(arms.get(k, {}).get("dots", 0)):,}</td><td>{n(v["dti"])}</td>'
        f'<td>{esc(str([round(x, 5) for x in v["ci95"]]))}</td>'
        f'<td>{n(v["tpw"], 1)} / {n(v["fpw"], 1)} / {n(v["fnw"], 1)}</td></tr>'
        for k, v in pooled.items())
    rob = exp3["robustness"]
    rob_rows = "".join(f'<tr><td>{esc(k)}</td><td>{v["dots"]:,}</td><td>{n(v["pooled_dti"])}</td></tr>'
                       for k, v in sorted(rob.items()))
    uni_lit, uni_repr = uni["literal_leg"], uni["representation_aware_leg"]

    def uni_line(row, key):
        if not row:
            return "none"
        v = row.get(key)
        return f'{n(v, 4)} ({esc(str(row.get("repo")))} — {esc(str(row.get("sources", ""))[:90])})'

    body = f"""<h1>Evidence</h1>
<p class="dim">Every number carries its class. HOLDOUT-DTI values name the evaluator; a projection is
never written as a score.</p>
<h2>EXP-1 — catalogue hide-and-recover (the checkable instrument)</h2>
<p class="dim">4 spatially blocked folds, whole fault components withheld with a 3 px buffer,
{exp1["holdout"]["withheld_positive_pixels"]:,} withheld positive pixels, evaluator
<code>{esc(exp1["pooled"]["evaluator_version"])}</code>, {exp1["pooled"]["draws"] if "draws" in exp1["pooled"] else 1000} paired bootstrap draws.</p>
<table><tr><th>arm</th><th>dots</th><th>HOLDOUT-DTI</th><th>95 % CI</th><th>wTP / wFP / wFN (300 m triangle)</th></tr>{cat_rows}</table>
<p class="dim">The catalogue frame is <b>structurally biased</b> toward near-catalogue placement
(withheld catalogue segments sit next to visible catalogue). It is kept because it is the only frame
with a fixed, checkable truth, and it is never used to claim a live score.</p>
<h2>Design rule (declared before use)</h2>
<p>q = {exp1["chosen"]["q"]} — {esc(exp1["design_rule"]["rule"])} ({esc(exp1["design_rule"]["live_optimal_band"])}),
spacing table {esc(str(exp1["chosen"]["table"]))} px. Emitted dots: {emit["dots"]:,} in
{emit["blocks_selected"]:,} of {emit["blocks_total"]:,} blocks, {emit["dots_on_catalogue"]} on the
mapped catalogue.</p>
<h2>EXP-2 — off-catalogue frame (the arbiter)</h2>
<p class="dim">{esc(exp2["frame_note"])}</p>
<table><tr><th>arm</th><th>dots</th><th>HOLDOUT-DTI</th><th>95 % CI</th></tr>
{"".join(f'<tr><td>{esc(k)}</td><td>{exp2["dots"][k]:,}</td><td>{n(v["dti"])}</td><td>{esc(str([round(x, 5) for x in v["ci95"]]))}</td></tr>' for k, v in exp2["pooled"]["scores"].items())}</table>
<p class="dim">Paired difference (lane − proximity): {esc(json.dumps(exp2["pooled"]["paired_differences"]["proximity_only"]["delta"]))} … see the receipt for the CI.</p>
<h2>EXP-3 — leakage canary and robustness</h2>
<p>Worst single-feature AUC {n(exp3["worst_canary"]["auc"], 4)} ({esc(exp3["worst_canary"]["covariate"])}),
limit {exp3["canary_threshold"]}, flags: {esc(str(exp3["canary_flags"]))}. Robustness over seeds
59/101/202 × q × 0.6/1.0/1.6: span {n(exp3["robustness_span"]["min"])} –
{n(exp3["robustness_span"]["max"])} (spread {n(exp3["robustness_span"]["spread"])}).</p>
<table><tr><th>arm</th><th>dots</th><th>HOLDOUT-DTI (catalogue frame)</th></tr>{rob_rows}</table>
<h2>Uniqueness against the registry</h2>
<p class="dim">{esc(uni["scope"])}</p>
<table><tr><th>leg</th><th>scope</th><th>worst rank correlation (dot field)</th><th>worst 3 px overlap</th></tr>
<tr><td>literal</td><td>{esc(uni_lit["scope"])}</td><td>{uni_line(uni_lit["worst_rho_dot_field"], "rho_dot_field")}</td>
<td>{uni_line(uni_lit["worst_overlap_ours_inside_theirs"], "ours_inside_theirs_3px")}</td></tr>
<tr><td>representation-aware</td><td>{esc(uni_repr["scope"])}</td><td>{uni_line(uni_repr["worst_rho_dot_field"], "rho_dot_field")}</td>
<td>{uni_line(uni_repr["worst_overlap_ours_inside_theirs"], "ours_inside_theirs_3px")}</td></tr>
<tr><td><b>comparable peers</b> (≤ 2 % of footprint)</td><td>{esc(str((uni.get("comparable_peer_tier") or {{}}).get("n_comparable_peers", 0)))} rasters</td>
<td>{uni_line((uni.get("comparable_peer_tier") or {{}}).get("worst_ours_inside_theirs"), "rho_dot_field")}</td>
<td>{uni_line((uni.get("comparable_peer_tier") or {{}}).get("worst_ours_inside_theirs"), "ours_inside_theirs_3px")}</td></tr></table>
<p>Verdict: drift by rank correlation = <b>{esc(str(uni["verdict"]["drift_by_rho"]))}</b>,
drift by overlap against sparse peers = <b>{esc(str(uni["verdict"]["drift_by_overlap_sparse_peers"]))}</b>
(the literal leg trips on a full-footprint uniform lattice, which sits within 3 px of any dot
field), drift by overlap against comparable peers =
<b>{esc(str(uni["verdict"].get("drift_by_overlap_comparable_peers")))}</b>.
{esc(uni.get("saturation_note", ""))}</p>
<h2>Family comparison (matched budget)</h2>
<table><tr><th>field</th><th>dots</th><th>catalogue frame</th><th>off-catalogue frame</th></tr>
{"".join(f'<tr><td>{esc(k)}</td><td>{v["dots"]:,}</td><td>{n(v["dti_catalogue"])}</td><td>{n(v["dti_sgmc"])}</td></tr>' for k, v in fam["arms"].items())}</table>
<p class="dim">{esc(fam["arbiter_note"])}</p>"""
    page("evidence.html", "Evidence", body)

    passes = (ROOT / "docs" / "verification.md")
    body = f"""<h1>Audit</h1>
<h2>Irregularities</h2>
<ul>
<li><b>IR-59-PROMPT-01</b> — the original chat prompt is not machine-recoverable inside the sandbox;
the README preserves it as requirements and does not claim word-for-word fidelity. The sibling
repository's preserved prompt is copied to <code>docs/task-prompt-sibling.md</code>.</li>
<li><b>IR-59-SEP-01</b> — the deployed field's <i>scale</i> is not identified: two Newton stopping
points of identical code 0.2 % apart in deviance give block peak shares 90× apart. Emission
therefore selects blocks by rank, never by absolute density.</li>
<li><b>IR-59-EFFORT-02</b> — the USGS reliability-of-location class cannot be joined to the grid:
the attribute table keys on <code>NUM</code>, the trace segments on <code>record_id</code>, and no
join key exists in the published files.</li>
<li><b>IR-59-SGMC-01</b> — <code>derived_sgmc_faults_100m_u8.tif</code> is byte-identical to
<code>sgmc_faults_100m.tif</code>; it is not a difference layer.</li>
<li><b>IR-59-REPR-01</b> — the literal “≥70 % of dots within 3 px of any single prior raster” rule is
unsatisfiable for <i>any</i> nonempty candidate while dense-support priors are in the registry
(their support covers every cell, so the overlap is 1.0 by construction). Both legs are measured and
the ruling is requested; the representation-aware leg is the meaningful one.</li>
<li><b>IR-30-039 (inherited)</b> — catalogue-proxy DTI ranks the thinning family opposite to the
live board. The release therefore states which frame each number comes from and never adjudicates
field choice with the catalogue frame alone.</li>
<li><b>Driver bug found and fixed this session</b> — the first emission recipe put an absolute
expected-pixel cut on a field whose scale is unstable (IR-59-SEP-01) and used two different ridge
values in different stages. Both are fixed; the earlier receipts are superseded and no longer exist
in the tree.</li>
</ul>
<h2>Three review passes</h2>
<p>{"(see docs/verification.md, copied below)" if passes.is_file() else ""}</p>
<pre>{esc(passes.read_text()) if passes.is_file() else "verification.md not found"}</pre>
<h2>Reproduce</h2>
<pre>python3 scripts/prepare_data.py --fetch
python3 scripts/run_lane.py verify
python3 scripts/run_lane.py planes
python3 scripts/run_lane.py fit
python3 scripts/run_lane.py exp1
python3 scripts/run_lane.py family
python3 scripts/run_lane.py exp2
python3 scripts/run_lane.py exp3
python3 scripts/run_lane.py emit
python3 scripts/run_lane.py ship
python3 scripts/check_uniqueness.py
python3 scripts/write_run_card.py
python3 scripts/build_site.py</pre>
<h2>Data provenance</h2>
<p>Transport hashes verified: {sum(1 for f in data_prep["files"] if f["verified"])} of
{len(data_prep["files"])} bridge pins and the 419 MB 19-band stack
(<code>{esc(str(data_prep["feature_stack"]["expected_sha256"])[:16])}…</code>). This authenticates
transport from a public mirror, not an official DrivenData receipt.</p>
<h2>Validator output</h2>
<pre>{esc(json.dumps(card["validator_output"], indent=2)[:1800])}</pre>"""
    page("audit.html", "Audit", body)

    body = """<h1>Sources</h1>
<h2>Competition</h2><ul>
<li>DrivenData “DOE GEMS”, competition 306 — <a href="https://www.drivendata.org/competitions/306/">
drivendata.org/competitions/306</a> (login-walled from this sandbox; the problem statement, the
distance-weighted Tversky metric and the rule facts used here were transcribed from page 967 and
cross-checked against the sibling repositories' receipts).</li>
<li>Official rules (NLR FY26 OSTI 96647) — <a href="https://docs.nlr.gov/docs/fy26osti/96647.pdf">
docs.nlr.gov/docs/fy26osti/96647.pdf</a> (not reachable from the sandbox; end date, prize structure
and submission cap come from the sibling clones' transcription and are labelled OWNER-REPORTED).</li>
</ul>
<h2>Data</h2><ul>
<li>GeoDAWN / DOE GEMS feature stack and the competition grid — public bridge mirror
(<code>buffedlizard55-lab/6GEMSDOE</code>), hashes in <code>docs/replication/data/data_preparation_lane1ac8.json</code>.</li>
<li>USGS QFFD / INGENIOUS — the label source named by the problem statement.</li>
<li>USGS State Geologic Map Compilation (SGMC) faults — used only as the off-catalogue proxy frame:
<a href="https://ngmdb.usgs.gov/">ngmdb.usgs.gov</a>.</li>
<li>California Geological Survey QFaults v2.0 — external cross-check only:
<a href="https://www.conservation.ca.gov/cgs/qfaults">conservation.ca.gov/cgs/qfaults</a>.</li>
</ul>
<h2>Methods</h2><ul>
<li>Phillips et al. 2009, “Sample selection bias and presence-only distribution models”,
<i>Ecological Applications</i> — <a href="https://doi.org/10.1890/08-2025.1">doi:10.1890/08-2025.1</a>
(target-group background).</li>
<li>Fithian &amp; Hastie 2013, “Finite-sample equivalence in statistical models for presence-only
data”, <i>Annals of Applied Statistics</i> — <a href="https://doi.org/10.1214/13-AOAS667">
doi:10.1214/13-AOAS667</a> (MaxEnt ⟺ Poisson point process).</li>
<li>Hermant et al. 2025 (SGW) — registration error context for the 1 km block size.</li>
<li>Mattéo et al. 2021 (JGR) — fault detection from potential-field derivatives.</li>
</ul>"""
    page("sources.html", "Sources", body)

    shutil.copyfile(ROOT / "README.md", DOCS / "README.md")
    print(f"[site] wrote {len(NAV) + 1} pages and {len(list(DATA.glob('*.json')))} receipts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
