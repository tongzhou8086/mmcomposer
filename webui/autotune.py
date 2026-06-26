"""Shim: this module moved to ``mmcomposer.autotune`` (Stage B migration).

Kept so existing in-repo ``import autotune`` -- and ``python webui/autotune.py
<shape>`` -- keep working.  Adds the repo root to sys.path, aliases to the
relocated module, and forwards the CLI when run as a script.
"""
import pathlib as _p
import sys as _s

_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))  # repo root
import mmcomposer.autotune as _moved  # noqa: E402

_s.modules[__name__] = _moved

if __name__ == "__main__":
    raise SystemExit(_moved.main())
