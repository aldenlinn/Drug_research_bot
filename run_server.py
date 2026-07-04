from __future__ import annotations

import os
import sys

# serve_api.py strips its own directory and the cwd from sys.path (a bitsandbytes-shadow guard) and
# never restores them, so a plain python serve_api.py cannot import the sibling local modules. Import
# them here first, while the repo is on sys.path, so they are cached in sys.modules before serve_api
# runs its strip; the cached modules then resolve regardless of the stripped path.
REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import Reasearch_Drug_Chatbot  # noqa: F401  cache before serve_api strips sys.path
import rag_retriever  # noqa: F401
import drugbot_prompts  # noqa: F401
import ie_client  # noqa: F401
import live_retriever  # noqa: F401

# Importing serve_api runs its module-level engine load (base model + adapter + retriever) and its
# sys.path strip; every local import it needs is already satisfied from the cache above.
import serve_api

if __name__ == "__main__":
    serve_api.main()
