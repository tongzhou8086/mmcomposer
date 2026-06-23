#!/usr/bin/env bash
# Build a self-contained HTML wrapper around the slides PDF and (re)upload it to
# hibe.dev. Re-running updates the same project (same name), so the public share
# URL stays stable.
#
#   ./upload_to_hibe.sh                # uses ./main.pdf, name "gemm-blackwell-slides"
#   ./upload_to_hibe.sh other.pdf      # upload a different PDF
#   HIBE_NAME=my-talk ./upload_to_hibe.sh
#
# Prereq: a hibe token at ~/.config/hibe/token. To mint one (device flow):
#   curl -s -X POST https://hibe.dev/api/auth/device           # note user_code + device_code
#   # open https://hibe.dev/device, enter the user_code, approve
#   curl -s -X POST https://hibe.dev/api/auth/device/poll \
#     -H 'content-type: application/json' -d '{"device_code":"<device_code>"}'
#   mkdir -p ~/.config/hibe && printf %s '<access_token>' > ~/.config/hibe/token && chmod 600 ~/.config/hibe/token

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF="${1:-$HERE/main.pdf}"
NAME="${HIBE_NAME:-gemm-blackwell-slides}"
HTML="$HERE/slides_hibe.html"
TOKEN_FILE="${HIBE_TOKEN_FILE:-$HOME/.config/hibe/token}"
TITLE="How to Design a High Performance GEMM Kernel (on Blackwell)?"

[ -f "$PDF" ] || { echo "ERROR: PDF not found: $PDF" >&2; exit 1; }
TOK="$(tr -d '[:space:]' < "$TOKEN_FILE" 2>/dev/null || true)"
[ -n "$TOK" ] || { echo "ERROR: empty/missing token at $TOKEN_FILE (see device-flow notes at top)." >&2; exit 1; }

# 1. Build HTML: embed the PDF as a base64 data: URI (inline viewer + download link).
python3 - "$PDF" "$HTML" "$TITLE" <<'PY'
import base64, sys
pdf, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
b = base64.b64encode(open(pdf, "rb").read()).decode()
html = (
    '<!doctype html><meta charset=utf-8><title>' + title + '</title>'
    '<body style=margin:0>'
    '<header style="padding:10px 16px;background:#2a3140;color:#e6e6e6;'
    'font-family:-apple-system,Segoe UI,Roboto,sans-serif">'
    '<b>' + title + '</b> &nbsp; '
    '<a style=color:#7fb0ff href=data:application/pdf;base64,' + b +
    ' download=gemm-blackwell.pdf>Download PDF</a></header>'
    '<embed type=application/pdf style="width:100%;height:calc(100vh - 48px)" '
    'src=data:application/pdf;base64,' + b + '#view=FitH>'
)
open(out, "w").write(html)
print("built %s (%.2f MB)" % (out, len(html) / 1e6))
PY

# 2. Upload (create or update by name).
resp="$(curl -sS -X POST https://hibe.dev/api/projects/single-html \
    -H "Authorization: Bearer $TOK" \
    -F "name=$NAME" -F "html=@$HTML")"
echo "upload: $resp"

id="$(printf '%s' "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)"
[ -n "$id" ] || { echo "ERROR: upload failed (no project id in response)." >&2; exit 1; }

# 3. Ensure public sharing is on, then print the stable share URL.
share="$(curl -sS -X PATCH "https://hibe.dev/api/projects/$id" \
    -H "Authorization: Bearer $TOK" -H 'content-type: application/json' \
    -d '{"share_enabled":true,"share_public":true}')"
sid="$(printf '%s' "$share" | python3 -c "import sys,json;print(json.load(sys.stdin).get('share_id',''))" 2>/dev/null || true)"
[ -n "$sid" ] && echo "PUBLIC URL: https://hibe.dev/s/$sid/" || echo "share: $share"
