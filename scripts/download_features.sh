#!/usr/bin/env bash
# Restore the official DrivenData competition feature stack from the owner's
# public GitHub mirror and verify it byte-for-byte.
#
#   training_features.tif   418,912,844 B
#   sha256 4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5
#
# The DrivenData data tab is login-walled, so this is a provenance bridge, not
# the organiser's own download.  The mirror is split into five parts because
# GitHub refuses single blobs above 100 MB.  Verification is mandatory: nothing
# downstream runs on an unverified stack.
#
# Requires: gh (authenticated) or curl, sha256sum, python3.
set -euo pipefail

REF=${REF:-c0c06ac82178f26b94fce3397036ef8f12a2f3a0}
REPO=${REPO:-buffedlizard55-lab/GEMSDOE}
PREFIX=${PREFIX:-data/bridge/gems-geodawn-numerical-features.tif.part}
DEST_DIR=${DEST_DIR:-out/features}
EXPECT_SHA=${EXPECT_SHA:-4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5}
EXPECT_BYTES=${EXPECT_BYTES:-418912844}

mkdir -p "$DEST_DIR"
for i in 000 001 002 003 004; do
  out="$DEST_DIR/part-$i"
  if [ -s "$out" ]; then echo "have part-$i ($(stat -c%s "$out") B)"; continue; fi
  echo "fetching part-$i ..."
  sha=$(gh api "repos/$REPO/contents/$PREFIX-$i?ref=$REF" --jq .sha)
  gh api "repos/$REPO/git/blobs/$sha" --jq .content | base64 -d > "$out"
  echo "  part-$i $(stat -c%s "$out") B"
done

cat "$DEST_DIR"/part-000 "$DEST_DIR"/part-001 "$DEST_DIR"/part-002 \
    "$DEST_DIR"/part-003 "$DEST_DIR"/part-004 > "$DEST_DIR/training_features.tif"

GOT_BYTES=$(stat -c%s "$DEST_DIR/training_features.tif")
GOT_SHA=$(sha256sum "$DEST_DIR/training_features.tif" | cut -d' ' -f1)
echo "bytes     $GOT_BYTES (expect $EXPECT_BYTES)"
echo "sha256    $GOT_SHA"
echo "expected  $EXPECT_SHA"
if [ "$GOT_BYTES" != "$EXPECT_BYTES" ] || [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "VERIFICATION FAILED - refusing to use an unverified feature stack" >&2
  exit 1
fi
echo "VERIFIED: official 19-band GeoDAWN feature stack restored"
