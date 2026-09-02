#!/bin/sh
# CDU demo launcher (macOS / Linux). English only - terminal locale varies.
# No absolute paths: everything is relative to this file's folder.
cd "$(dirname "$0")" || exit 1

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=python

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null || {
  echo "Python 3.12 or newer is required. Install it, then run this file again."
  exit 1
}

echo "Serving this folder on http://localhost:8000/ - press Ctrl+C to stop."
URL=http://localhost:8000/pfd.html
(sleep 1; open "$URL" 2>/dev/null || xdg-open "$URL" 2>/dev/null) &
exec "$PY" -m http.server 8000
