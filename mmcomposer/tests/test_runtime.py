#!/usr/bin/env python3
"""Unit tests for the runtime module (webui/runtime.py).

Two layers:
  * launch_dims equivalence -- PURE, no GPU: assert runtime.launch_dims matches
    the authoritative gpu_codegen_driver.launch_spec across many valid combos
    and shapes (no hard-coded magic numbers).
  * correctness -- GPU: compile a known-good config and assert kernel()(a,b)
    matches torch.matmul.  Skipped when CUDA is unavailable.

Run:  python webui/tests/test_runtime.py   (or pytest)
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webui" / "tests"))  # gpu_codegen_driver harness

from mmcomposer import mvp_core as mc
from mmcomposer import combos
from mmcomposer import runtime

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 4608, 768)]
NUM_SMS = 132  # fixed so the persistent-grid branch is exercised deterministically


def test_launch_dims_matches_driver_launch_spec():
    import gpu_codegen_driver as d  # imports torch; fine on a CPU node
    ws = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    filt = {"bn": [256, 512], "ns": [3, 4, 6], "single_tmem_policy": "all"}
    n = 0
    for tier, k in combos.valid_combos(ws, filt):
        cfg = runtime.config_from_combo(tier, k)
        for (M, N, K) in SHAPES:
            got = runtime.launch_dims(cfg, M, N, K, num_sms=NUM_SMS)
            want = d.launch_spec(tier, k, M, N, K, num_sms=NUM_SMS)
            assert got == want, f"mismatch {tier['dir']} {k} @ {M}x{N}x{K}: {got} != {want}"
            n += 1
    assert n > 0
    print(f"    ({n} combo x shape launch-dim comparisons)")


def test_config_from_combo_has_symbol_and_cluster():
    ws = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
    tier, k = next(iter(combos.valid_combos(ws, {"bn": [256], "ns": [4]})))
    cfg = runtime.config_from_combo(tier, k)
    assert cfg["symbol"] == tier["symbol"]
    assert cfg["cluster"] == tier["cluster"]
    assert cfg["bn"] == k["bn"]


def test_correctness_matches_torch():
    import torch
    if not torch.cuda.is_available():
        print("    SKIP (no CUDA)")
        return
    from mmcomposer import compiler

    # A known-good 2-CTA overlapped/TMA-pipelined config (validated combo).
    tier = mc.TIER_MAP[(True, True)]
    k = dict(bm=128, bn=256, bk=64, ns=4, gsm=8, nw=8, persistent=1, ld_width=8,
             overlap=1, split_epilogue=0, l1_no_alloc=0, tma_pipelined=1,
             tma_store_stages=2, single_tmem=0)
    assert combos.is_valid(tier, k), "test config must be valid"

    M = N = K = 1024  # 2-CTA needs M % (2*BM)=256 == 0, N % BN=256 == 0, K % BK=64 == 0
    with tempfile.TemporaryDirectory() as dtmp:
        src = pathlib.Path(dtmp) / "kernel.cu"
        src.write_text(mc.render_kernel(
            tier, k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
            ld_width=k["ld_width"], overlap=k["overlap"],
            split_epilogue=k["split_epilogue"], l1_no_alloc=k["l1_no_alloc"],
            tma_pipelined=k["tma_pipelined"], tma_store_stages=k["tma_store_stages"],
            single_tmem=k["single_tmem"]))
        cubin = compiler.compile_one(str(src))

        cfg = runtime.config_from_combo(tier, k)
        gemm = runtime.kernel(cfg, cubin)
        a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        c = gemm(a, b)
        ref = (a.float() @ b.float())
        rel = (c.float() - ref).norm() / ref.norm()
        print(f"    rel err = {rel.item():.4e}")
        assert rel.item() < 0.05, f"rel err too high: {rel.item()}"


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
