#!/usr/bin/env bash
# Daily local backup of ArgusOrb SQLite database
set -euo pipefail

REMOTE="root@76.13.118.240"
REMOTE_DB="/opt/argus/data/starlink.db"
LOCAL_DIR="/Users/yong/projects/argus/backups"
DATE=$(date +%Y%m%d)

mkdir -p "$LOCAL_DIR"
ssh "$REMOTE" "sqlite3 $REMOTE_DB '.backup /tmp/starlink-backup.db'" 2>/dev/null
scp -q "$REMOTE:/tmp/starlink-backup.db" "$LOCAL_DIR/starlink-$DATE.db"
ssh "$REMOTE" "rm -f /tmp/starlink-backup.db"

SIZE=$(ls -lh "$LOCAL_DIR/starlink-$DATE.db" | awk '{print $5}')
echo "[BACKUP] $DATE: $SIZE → $LOCAL_DIR/starlink-$DATE.db"
