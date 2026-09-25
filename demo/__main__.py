"""Launcher so the demo is one command with no uvicorn flags to remember.

    python -m demo
    python -m demo --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the weathering-chamber AI demo UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # Detectors are fitted on startup (~10 s); give that a head start so the
        # first page load is not a blank screen.
        threading.Timer(11.0, lambda: webbrowser.open(url)).start()
    print(f"Starting demo on {url} (Ctrl+C to stop)")
    uvicorn.run("demo.app:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
