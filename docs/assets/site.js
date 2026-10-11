/* Fills the status strip from the run card and the leaderboard feed.
   Every request is a URL relative to this page, so the same file works on the
   public site and in a sandboxed preview without pointing at any fixed host. */
(function () {
  function text(el, value) { if (el) { el.textContent = value; } }
  function fmt(n) { return (n === null || n === undefined) ? 'n/a' : Number(n).toLocaleString('en-US'); }
  fetch('data/run_card.json').then(function (r) { return r.json(); }).then(function (card) {
    text(document.getElementById('live-verdict'), card.verdict);
    text(document.getElementById('live-file'), (card.file || '').split('/').pop());
    text(document.getElementById('live-sha'), card.raster_sha256);
    text(document.getElementById('live-emitted'), fmt(card.emitted));
    text(document.getElementById('live-holdout'),
         Number(card.holdout_dti.dti).toFixed(5) + ' [' + Number(card.holdout_dti.ci95[0]).toFixed(5) +
         ', ' + Number(card.holdout_dti.ci95[1]).toFixed(5) + ']');
    text(document.getElementById('live-projection'), card.projection.band);
    text(document.getElementById('live-slots'), fmt(card.submission_slots_used));
    var zip = document.getElementById('live-zip'); if (zip && card.zip_file) { zip.href = card.zip_file; }
    var tif = document.getElementById('live-tif'); if (tif && card.file) { tif.href = card.file; }
  }).catch(function () { text(document.getElementById('live-verdict'), 'run card unavailable'); });
  fetch('data/leaderboard.json').then(function (r) { return r.json(); }).then(function (board) {
    var top = board.rows && board.rows[0];
    text(document.getElementById('live-board-top'), top ? (top.participant + ' ' + top.best_public_dw_tversky) : 'n/a');
    text(document.getElementById('live-board-fetched'), board.fetched + ' rev ' + board.revision);
  }).catch(function () {});
}());
