#!/usr/bin/env bash
# Upload the interview deck to Google Drive using the gcloud user credential.
#
# Requires Drive scope on the credential:
#   gcloud auth login --enable-gdrive-access
#
# The file is streamed from disk by curl, so its size is not a constraint.
# Uploaded as .pptx WITHOUT conversion: Google Slides has no Avenir Next and would
# substitute Arial in the headings while the figures keep their baked-in type.
set -euo pipefail

DECK="${1:-results/slides/video_pipeline_deck.pptx}"
TITLE="${2:-Shot, Face, Speaker — video pipeline}"
PPTX_MIME="application/vnd.openxmlformats-officedocument.presentationml.presentation"

[ -f "$DECK" ] || { echo "no such file: $DECK" >&2; exit 1; }

TOKEN="$(gcloud auth print-access-token)"

# fail early and clearly if the credential still lacks Drive scope
code="$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  'https://www.googleapis.com/drive/v3/about?fields=user(emailAddress)')"
if [ "$code" != "200" ]; then
  echo "Drive API returned HTTP $code — the credential is missing the Drive scope." >&2
  echo "Run: gcloud auth login --enable-gdrive-access" >&2
  exit 2
fi

curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "metadata={\"name\":\"${TITLE}\",\"mimeType\":\"${PPTX_MIME}\"};type=application/json;charset=UTF-8" \
  -F "file=@${DECK};type=${PPTX_MIME}" \
  'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,size,webViewLink'
echo
