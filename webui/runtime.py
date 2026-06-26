"""Shim: this module moved to ``mmcomposer.runtime`` (Stage B migration).

Kept so existing in-repo ``import runtime`` keeps working while consumers are
repointed.  Adds the repo root to sys.path so ``mmcomposer`` is importable
regardless of cwd, then aliases this module to the relocated one.
"""
import pathlib as _p
import sys as _s

_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))  # repo root
import mmcomposer.runtime as _moved  # noqa: E402

_s.modules[__name__] = _moved
