#!/usr/bin/env bash
# Fetch the public GEMS data files that can be downloaded WITHOUT a DrivenData login, into data/.
#
# Official sources (fetched 2026-10-11 from the competition data page / problem page):
#   https://www.drivendata.org/competitions/306/competition-doe-gems/data/      (LOGIN REQUIRED)
#   https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/   (problem description)
#   Dropbox links listed on the competition's "data" section, as supplied by the project owner.
#
# DrivenData files (training_features.tif, labels.tif, sample_submission.tif, 1m_DEM_links.csv)
# require a logged-in browser session. This script cannot fetch them and says so; place them in
# data/ by hand (see docs/index.html#data).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

# Dropbox "dl=0" share links rewritten to direct-download "dl=1" form. Keys are the local filenames.
declare -A URLS=(
  ["example_submission.tif"]="https://www.dropbox.com/scl/fi/6rgvnuady818ol8yqgis4/example_submission.tif?rlkey=kbykilvau066xuogoosbf4cq8&st=8junzdyw&dl=1"
  ["existing_faults.tif"]="https://www.dropbox.com/scl/fi/t7fyt03qdh9egyme0itwo/existing_faults.tif?rlkey=yiao96uluqdkipf0h5vju71jf&st=rnino7ya&dl=1"
  ["gems-geodawn-numerical-features.tif"]="https://www.dropbox.com/scl/fi/3vz9o0wwavi26xaeoxlwr/gems-geodawn-numerical-features.tif?rlkey=je8d8fepqfbst9lnwsq9rkplu&st=zj1lag1r&dl=1"
  ["Digital-elevation-model-links-JSON.pdf"]="https://www.dropbox.com/scl/fi/ig0mban712ns1atphgphe/Digital-elevation-model-links-JSON.pdf?rlkey=zm77f1vbtt2if8hlruymptnu3&st=srhhir10&dl=1"
  ["GEMS_96647.pdf"]="https://www.dropbox.com/scl/fi/aemhtutjgcp6tr3tint94/GEMS_96647.pdf?rlkey=rek210cj2smnmzb8n0sla1vmd&st=wz4kofki&dl=1"
)

status=0
for name in "${!URLS[@]}"; do
  dest="data/$name"
  if [[ -s "$dest" ]]; then
    echo "[have] $name ($(stat -c %s "$dest") bytes)"
    continue
  fi
  echo "[get ] $name"
  if curl -fsSL --retry 3 --connect-timeout 20 -o "$dest.part" "${URLS[$name]}"; then
    mv "$dest.part" "$dest"
    echo "[ok  ] $name ($(stat -c %s "$dest") bytes)"
  else
    rm -f "$dest.part"
    echo "[FAIL] $name  (network blocked or link changed; download it in a browser instead)"
    status=1
  fi
done

echo
echo "Still required from DrivenData (login needed; place in data/ manually):"
for f in training_features.tif labels.tif sample_submission.tif 1m_DEM_links.csv; do
  if [[ -s "data/$f" ]]; then echo "  [have] $f"; else echo "  [MISSING] $f"; fi
done
exit $status
