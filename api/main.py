"""Entry point for Railway deployment."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from decimus.web.app import app  # noqa: F401, E402
