"""Re-derive the emission utility rule on SYNTHETIC grids (no competition data needed).

Two checks, both written to evidence/utility_table_synthetic.json:

1. EXACT SIGN IDENTITY. For random prediction/truth pairs and a random added dot, compare
   sign(DTI_after - DTI_before) with sign(dTP - alpha * DTI * dS), dS = dTP + dFP.
   This is an identity of the official formula, so it must hold in every trial.

2. SPACING SWEEP. A straight 60-px truth line (synthetic). Dots of probability 1 are placed every
   s px along a line offset by o px from the truth. We report DTI for each (s, o). This is only
   a toy check of the "about 3 px when dense" claim; it is NOT a holdout result.

Run:  PYTHONPATH=src python scripts/utility_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from gems_dti.metric import ALPHA, dti

OUT = Path(__file__).resolve().parents[1] / "evidence" / "utility_table_synthetic.json"


def identity_trials(n: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    agree = 0
    checked = 0
    for _ in range(n):
        h, w = 30, 30
        truth = rng.random((h, w)) < 0.02
        if truth.sum() == 0:
            continue
        pred = np.where(rng.random((h, w)) < 0.05, rng.random((h, w)), 0.0)
        y, x = int(rng.integers(h)), int(rng.integers(w))
        if pred[y, x] >= 1.0:
            continue
        before = dti(pred, truth)
        cand = pred.copy()
        cand[y, x] = 1.0
        after = dti(cand, truth)
        d_tp = after.tp_w - before.tp_w
        d_fp = after.fp_w - before.fp_w
        d_s = d_tp + d_fp
        lhs = after.dti - before.dti          # actual change in DTI
        rhs = d_tp - ALPHA * before.dti * d_s  # predicted sign from the exact rule
        checked += 1
        if abs(lhs) < 1e-12 or abs(rhs) < 1e-12 or np.sign(lhs) == np.sign(rhs):
            agree += 1  # ties carry no sign information
    return {"trials": checked, "sign_agreements": agree, "all_agree": agree == checked}


def spacing_sweep() -> dict:
    """Dots of p=1 every s px along a horizontal truth line, offset o px perpendicular to it."""
    h, w = 60, 120
    truth = np.zeros((h, w), dtype=bool)
    truth[30, 20:80] = True  # synthetic 60-px horizontal fault
    results = {}
    for offset in (0, 1, 2):
        row = {}
        for s in range(1, 9):
            pred = np.zeros((h, w))
            for x in range(20, 80, s):
                pred[30 - offset, x] = 1.0
            row[str(s)] = round(dti(pred, truth).dti, 4)
        results[f"perpendicular_offset_px={offset}"] = row
    return results


def main() -> None:
    out = {
        "label": "SYNTHETIC - not a competition or holdout score",
        "alpha": ALPHA,
        "identity_check": identity_trials(),
        "spacing_sweep_dti_by_spacing_px": spacing_sweep(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
