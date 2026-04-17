#!/usr/bin/env python3
"""Launch the Decimus web UI."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"\n  Decimus API → http://localhost:{port}\n")
    uvicorn.run(
        "decimus.web.app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=["decimus"],
    )
