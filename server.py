"""Talent Sphere Elevate — Flask entry point.

Development:  python server.py            (http://localhost:5000)
Production :  waitress-serve --listen=0.0.0.0:5000 server:app
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from webapp import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    print(f"\n  Talent Sphere Elevate -> http://{host}:{port}\n")
    try:
        from waitress import serve

        # threads: chat streaming holds a worker per active response.
        serve(app, host=host, port=port, threads=12)
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)
