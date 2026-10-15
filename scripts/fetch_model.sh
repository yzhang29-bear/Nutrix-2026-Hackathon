#!/usr/bin/env bash
# Download a model version from S3 into the Docker build context and verify every file against manifest.json.
#   ./scripts/fetch_model.sh [version]     # default: $MODEL_VERSION or v7-20261010
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-${MODEL_VERSION:-v7-20261010}}"
DEST="$ROOT/model/deploy/model_store/$VERSION"

aws --profile "${AWS_PROFILE:-nutrix}" s3 sync "s3://nutrix-models/models/$VERSION/" "$DEST/" --only-show-errors

python -I - "$DEST" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
files = json.loads((d / "manifest.json").read_text())["files"]
bad = [f for f, h in files.items() if not (d / f).exists() or hashlib.sha256((d / f).read_bytes()).hexdigest() != h]
if bad:
    sys.exit(f"checksum mismatch or missing: {bad}")
print(f"{d.name}: {len(files)} files verified")
EOF
