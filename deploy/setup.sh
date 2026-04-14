#!/usr/bin/env bash
# ArgusOrb — one-shot deploy script
# Usage: ssh ubuntu@<IP> 'bash -s' < deploy/setup.sh

set -euo pipefail

REPO="https://github.com/argusorb-labs/argus.git"
APP_DIR="/opt/argus"

echo "=== [1/4] Install Docker ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
fi

echo "=== [2/4] Clone repo ==="
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

mkdir -p "$APP_DIR/data"

echo "=== [3/4] Build and start ==="
docker compose up -d --build

echo "=== [4/4] Verify ==="
sleep 5
docker compose ps
curl -sf http://localhost:8000/api/status && echo " <- API OK"

echo ""
echo "Done. Site: https://argusorb.io"
