"""CLI: emit a specialized kernel/host from a config JSON.

    python -m codegen kernel config.json      # specialized kernel.cu -> stdout
    python -m codegen host   config.json      # specialized host.py   -> stdout

config.json is a flat object of the selected option values, e.g.
    {"skeleton": "tier3_cluster_swizzle", "BM": 128, "BN": 256, "BK": 64,
     "NS": 5, "GROUP_SIZE_M": 4, "NUM_WARPS": 8, "TMA_STORE": 0,
     "TCGEN05_LD_WIDTH": 8, "EPILOGUE_OVERLAP": 1, "EPILOGUE_SPLIT": 1,
     "label": "Tier 3 — + 2-CTA cluster MMA"}
"""

from __future__ import annotations

import json
import sys

from . import generate_host, generate_kernel


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("kernel", "host"):
        sys.stderr.write(__doc__ or "")
        return 2
    what, config_path = argv
    with open(config_path) as f:
        config = json.load(f)
    out = generate_kernel(config) if what == "kernel" else generate_host(config)
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
