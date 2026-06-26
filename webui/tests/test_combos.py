#!/usr/bin/env python3
"""Unit tests for the combo-enumeration module (webui/combos.py).

Pure / GPU-free -- run directly:  python webui/tests/test_combos.py
(or under pytest).  Demonstrates that `enumerate` can be exercised in complete
isolation: set up inputs, call the module, assert on the returned combos.
"""
import pathlib
import sys

WEBUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEBUI))

import mvp_core as mc
import combos

WS_DIRS = list(dict.fromkeys(t["dir"] for k, t in mc.TIER_MAP.items() if t and k[0]))
ALL_DIRS = list(dict.fromkeys(t["dir"] for t in mc.TIER_MAP.values() if t))

PROD = {"bn": [256, 512], "ns": [3, 4, 5, 6, 7], "two_cta": [1],
        "tma_store_stages": [1, 2], "single_tmem_policy": "bn512-only"}
FULL = {"single_tmem_policy": "all"}


def test_production_count():
    # The pruned production sweep -- the 738 combos referenced in the talk.
    assert sum(1 for _ in combos.valid_combos(WS_DIRS, PROD)) == 738


def test_full_count():
    assert sum(1 for _ in combos.valid_combos(ALL_DIRS, FULL)) == 10062


def test_validity_prunes_the_raw_grid():
    raw = sum(1 for _ in combos.all_combos(WS_DIRS, PROD))
    valid = sum(1 for _ in combos.valid_combos(WS_DIRS, PROD))
    assert raw == 8640
    assert valid == 738
    assert valid < raw  # validation actually drops combos


def test_valid_combos_consistent_with_is_valid():
    # Every combo from valid_combos must pass is_valid, and the count must equal
    # filtering all_combos through is_valid -- the two APIs can't disagree.
    n_via_filter = 0
    for tier, k in combos.all_combos(WS_DIRS, PROD):
        if combos.is_valid(tier, k):
            n_via_filter += 1
    n_via_valid = 0
    for tier, k in combos.valid_combos(WS_DIRS, PROD):
        assert combos.is_valid(tier, k)
        n_via_valid += 1
    assert n_via_filter == n_via_valid


def test_filters_restrict_dimensions():
    # Pinning bn=[256] must exclude bn=512 from the raw grid entirely.
    for _, k in combos.all_combos(WS_DIRS, {**PROD, "bn": [256]}):
        assert k["bn"] == 256


def test_tight_filter_yields_single_combo():
    one = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8], "two_cta": [1],
           "persistent": [1], "overlap": [1], "split_epilogue": [0],
           "l1_no_alloc": [0], "tma_pipelined": [1], "tma_store_stages": [2],
           "single_tmem": [0]}
    vc = list(combos.valid_combos(WS_DIRS, one))
    assert len(vc) == 1
    tier, k = vc[0]
    assert tier["dir"] == "tier3_cluster_swizzle"
    assert (k["bn"], k["ns"], k["tma_store_stages"]) == (256, 4, 2)


def test_is_valid_accepts_good_rejects_bad():
    tier3 = mc.TIER_MAP[(True, True)]  # warp-spec + 2-CTA cluster
    good = dict(bm=128, bn=256, bk=64, ns=4, gsm=8, nw=8, persistent=1,
                ld_width=8, overlap=1, split_epilogue=0, l1_no_alloc=0,
                tma_pipelined=1, tma_store_stages=2, single_tmem=0)
    assert combos.is_valid(tier3, good)
    assert not combos.is_valid(tier3, dict(good, ns=1))  # NS=1 below pipeline min


KNOB_KEYS = {"bm", "bn", "bk", "ns", "gsm", "nw", "persistent", "ld_width",
             "overlap", "split_epilogue", "l1_no_alloc", "tma_pipelined",
             "tma_store_stages", "single_tmem"}


def test_no_duplicate_valid_combos():
    # Enumeration must not emit the same combo twice.  A combo's identity is
    # (tier, knobs) -- and the single-CTA vs 2-CTA arms share a dir AND knobs
    # dict, distinguished only by tier["cluster"], so the signature must include
    # it (omitting it is what falsely flags the two arms as duplicates).
    seen = set()
    for tier, k in combos.valid_combos(ALL_DIRS, FULL):
        sig = (tier["dir"], tier["cluster"], tuple(sorted(k.items())))
        assert sig not in seen, f"duplicate combo: {sig}"
        seen.add(sig)


def test_every_combo_carries_full_knob_set():
    for _, k in combos.all_combos(WS_DIRS, PROD):
        assert set(k) == KNOB_KEYS


def test_filters_honored_across_dimensions():
    # Every pinned dimension must hold for every yielded combo (not just bn).
    f = {"bn": [256], "ns": [4, 5], "gsm": [8], "nw": [8]}
    for _, k in combos.all_combos(WS_DIRS, f):
        assert k["bn"] == 256
        assert k["ns"] in (4, 5)
        assert k["gsm"] == 8
        assert k["nw"] == 8


def test_two_cta_filter_selects_the_right_arms():
    small = {"bn": [256], "ns": [4], "gsm": [8], "nw": [8]}
    single = list(combos.all_combos(ALL_DIRS, {**small, "two_cta": [0]}))
    cluster = list(combos.all_combos(ALL_DIRS, {**small, "two_cta": [1]}))
    assert single and cluster
    assert all(not tier["cluster"] for tier, _ in single)
    assert all(tier["cluster"] for tier, _ in cluster)


def test_unknown_two_cta_yields_empty():
    assert list(combos.all_combos(ALL_DIRS, {"two_cta": [7]})) == []


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
