#!/usr/bin/env bash
# Delete .ts segments older than 7 days. Runs daily via launchd.
set -euo pipefail
CAM_ID="${1:-fd80836d}"
DIR="$HOME/recordings/$CAM_ID/segments"
[ -d "$DIR" ] || exit 0
find "$DIR" -type f -name "*.ts" -mtime +7 -delete
# Rebuild master m3u8 from remaining segments (simple regen — don't append forever)
echo "[cleanup] $(date) — segments remaining: $(ls -1 $DIR/*.ts 2>/dev/null | wc -l)"
