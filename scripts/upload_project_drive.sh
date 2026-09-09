#!/usr/bin/env bash
# Upload exactly one private migration bundle, only to the intended Google account.
set -euo pipefail
BUNDLE=${1:?Usage: bash scripts/upload_project_drive.sh /path/bundle.tar [remote-name]}
REMOTE=${2:-huydrive}
[[ "$REMOTE" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use a configured remote NAME, without colon/path'; exit 2; }
[[ -f "$BUNDLE" && "$BUNDLE" == *.tar ]] || { echo 'Missing .tar bundle'; exit 2; }
command -v rclone >/dev/null || { echo 'Install rclone, then: rclone config'; exit 2; }
# Official: https://rclone.org/commands/rclone_config_userinfo/
rclone config userinfo "$REMOTE:" --json | python3 -c '
import json,sys
d=json.load(sys.stdin)
emails=[str(v).lower() for k,v in d.items() if "email" in k.lower()]
if "huyphan1610@gmail.com" not in emails:
    sys.exit("STOP: remote is not authenticated as huyphan1610@gmail.com")
print("Verified Google Drive account: huyphan1610@gmail.com")'
DEST="$REMOTE:AAAI26-HUG-migration/$(basename "$BUNDLE")"
# Immutable prevents replacing a different existing backup; never use sync/delete.
rclone copyto "$BUNDLE" "$DEST" --immutable --checksum --progress
rclone check "$(dirname "$BUNDLE")" "$REMOTE:AAAI26-HUG-migration" \
  --include "/$(basename "$BUNDLE")" --one-way
echo "Uploaded and checked: $DEST"
