#!/bin/bash
# Full-page screenshot of the built site through a local server (fetch() of demo_data.json needs http, not file://)
# usage: screenshot_site.sh <site dir> <out.png> [width] [height]
SITE=$1; OUT=$2; W=${3:-1440}; H=${4:-9000}; PORT=8765
python3 -m http.server $PORT --bind 127.0.0.1 --directory "$SITE" >/dev/null 2>&1 &
SRV=$!
sleep 1
google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --window-size=$W,$H --virtual-time-budget=15000 \
  --screenshot="$OUT" "http://127.0.0.1:$PORT/index.html" >/dev/null 2>&1
kill $SRV
echo "saved $OUT"
