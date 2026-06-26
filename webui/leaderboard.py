"""Shim: this module moved to ``mmcomposer.leaderboard`` (Stage B migration).

Kept so existing in-repo ``import leaderboard`` keeps working while consumers are
repointed.  Adds the repo root to sys.path so ``mmcomposer`` is importable
regardless of cwd, then aliases this module to the relocated one.
"""
import pathlib as _p
import sys as _s

_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))  # repo root
import mmcomposer.leaderboard as _moved  # noqa: E402

_s.modules[__name__] = _moved
