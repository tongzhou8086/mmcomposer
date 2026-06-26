"""Tuning-results cache --- the `cache` module from DESIGN.md.

Stores, per shape `(M, N, K, dtype, arch)`, the ranked configs found by tuning:
`put` a result, `get`/`top_n`/`best` to read.  Pure / no GPU.

Two design points from DESIGN.md:
  * **Pluggable backend.**  The default is local disk
    (`~/.cache/mmcomposer/results/<key>.json`); a future remote/network backend
    implements the same `load`/`store` interface so a tune by anyone on the same
    arch is reusable everywhere -- no call-site changes.
  * This is the *config/results* cache (tiny JSON), NOT the cubin artifact cache
    (the `compile` module owns that on disk, arch-specific, local-only).

A *record* is any dict with at least ``config`` (the kernel config) and
``tflops``; the cache dedups by ``config`` and keeps records sorted by ``tflops``.
"""
from __future__ import annotations

import json
import os
import pathlib

DEFAULT_DTYPE = "bf16"
DEFAULT_ARCH = "sm_100a"


def cache_root() -> pathlib.Path:
    """Root cache dir: $MMCOMPOSER_CACHE_DIR, else $XDG_CACHE_HOME/mmcomposer,
    else ~/.cache/mmcomposer."""
    env = os.environ.get("MMCOMPOSER_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "mmcomposer"


def shape_key(M, N, K, dtype: str = DEFAULT_DTYPE, arch: str = DEFAULT_ARCH) -> str:
    """The cache key for a shape: e.g. '4096x4096x4096_bf16_sm_100a'."""
    return f"{M}x{N}x{K}_{dtype}_{arch}"


class LocalDiskBackend:
    """Default backend: one JSON file per shape key under <root>/results/."""

    def __init__(self, root=None):
        self.root = pathlib.Path(root) if root is not None else cache_root()

    def _path(self, key: str) -> pathlib.Path:
        return self.root / "results" / f"{key}.json"

    def load(self, key: str) -> list:
        p = self._path(key)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001  -- a corrupt/half-written file reads as empty
            return []

    def store(self, key: str, records: list) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(records, f)
        os.replace(tmp, p)   # atomic publish

    def keys(self) -> list:
        d = self.root / "results"
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob("*.json"))


class Cache:
    """Ranked results keyed by shape, backed by a pluggable store."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else LocalDiskBackend()

    @staticmethod
    def _sig(config) -> str:
        return json.dumps(config or {}, sort_keys=True)

    def put(self, key: str, record: dict) -> dict:
        """Add/replace a result for `key` (dedup by config), keep sorted by tflops."""
        if "config" not in record or "tflops" not in record:
            raise ValueError("record must have 'config' and 'tflops'")
        sig = self._sig(record["config"])
        recs = [r for r in self.backend.load(key) if self._sig(r.get("config")) != sig]
        recs.append(record)
        recs.sort(key=lambda r: (r.get("tflops") if r.get("tflops") is not None else -1.0),
                  reverse=True)
        self.backend.store(key, recs)
        return record

    def get(self, key: str) -> list:
        """All records for `key` (ranked), or [] if none."""
        return self.backend.load(key)

    def top_n(self, key: str, n: int) -> list:
        return self.backend.load(key)[:n]

    def best(self, key: str):
        """Top-ranked record for `key`, or None."""
        recs = self.backend.load(key)
        return recs[0] if recs else None

    def clear(self, key: str) -> None:
        recs = self.backend.load(key)
        if recs:
            self.backend.store(key, [])

    def keys(self) -> list:
        return self.backend.keys() if hasattr(self.backend, "keys") else []


# ---- module-level default cache (local disk) ------------------------------
_default = Cache()


def put(key, record):
    return _default.put(key, record)


def get(key):
    return _default.get(key)


def top_n(key, n):
    return _default.top_n(key, n)


def best(key):
    return _default.best(key)
