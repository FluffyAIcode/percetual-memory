from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
import os
from pathlib import Path


_DEFAULT_DIALECT_PATH = Path(
    "/Users/Allen/cursor/bayes/third_party/mempalace_official/dialect.py"
)
_OFFICIAL_DIALECT_PATH = Path(
    os.environ.get("MEMPALACE_OFFICIAL_DIALECT_PATH", str(_DEFAULT_DIALECT_PATH))
)

if not _OFFICIAL_DIALECT_PATH.exists():
    raise FileNotFoundError(
        f"Official MemPalace dialect source not found: {_OFFICIAL_DIALECT_PATH}"
    )

_loader = SourceFileLoader(
    "_mempalace_official_dialect", str(_OFFICIAL_DIALECT_PATH)
)
_spec = importlib.util.spec_from_loader("_mempalace_official_dialect", _loader)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load official dialect module from {_OFFICIAL_DIALECT_PATH}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

Dialect = _module.Dialect
