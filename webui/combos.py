"""Combo enumeration --- the `enumerate` module from DESIGN.md.

Pure, no GPU.  Expands the knob grid (restricted by `filters`) and decides which
combos are valid.  Shared by the codegen driver, autotune, and (later) the mmc
package, so "what is a valid combo" lives in exactly one place.
"""
import itertools

import mvp_core as mc


def _opts(filters, name, default):
    return filters.get(name) or default


def all_combos(tier_dirs, filters=None):
    """Yield (tier, knobs-dict) over the knob grid, restricted by `filters`.

    This is the RAW grid --- validity is not applied here (see `is_valid` /
    `valid_combos`).  `filters` restricts selected knob dimensions for pruned
    sweeps; with no filters it yields every grid point.
    """
    filters = filters or {}
    bn_list = _opts(filters, "bn", mc.BN_OPTS)
    ns_list = _opts(filters, "ns", mc.NS_OPTS)
    gsm_list = _opts(filters, "gsm", mc.GSM_OPTS)
    nw_list = _opts(filters, "nw", mc.NW_OPTS)
    ld_list = _opts(filters, "ld_width", mc.TCGEN05_LD_WIDTH_OPTS)
    l1_list = _opts(filters, "l1_no_alloc", mc.EPILOGUE_L1_NO_ALLOC_OPTS)
    two_cta_list = filters.get("two_cta")
    tma_filter = filters.get("tma_pipelined")
    tma_store_stage_filter = filters.get("tma_store_stages")
    single_filter = filters.get("single_tmem")
    single_tmem_policy = filters.get("single_tmem_policy")
    # A skeleton dir may back >1 tier: the warp-spec single-CTA and 2-CTA
    # cluster tiers share one dir, distinguished by the TWO_CTA knob.  Sweep
    # every (ms_ws, two_cta) arm registered for each requested dir.
    keys_for_dir = {}
    for _key, t in mc.TIER_MAP.items():
        if t:
            keys_for_dir.setdefault(t["dir"], []).append(_key)
    tier_keys = [k for tdir in tier_dirs for k in keys_for_dir[tdir]
                 if two_cta_list is None or int(k[1]) in two_cta_list]
    for key in tier_keys:
        tier = mc.TIER_MAP[key]
        # PERSISTENT is a launch knob (same cubin) --- only the persistent-
        # capable tiers get the grid=#SMs variant; others stay at [0].
        pers_default = mc.PERSISTENT_OPTS if tier.get("persistent_ok") else [0]
        pers_opts = _opts(filters, "persistent", pers_default)
        # EPILOGUE_OVERLAP only applies on the persistent-capable path; most
        # overlap=1 combos are filtered by the validator (persistent/NW/SMEM).
        ov_default = mc.EPILOGUE_OVERLAP_OPTS if tier.get("persistent_ok") else [0]
        ov_opts = _opts(filters, "overlap", ov_default)
        # EPILOGUE_SPLIT is a Tier 3 cluster epilogue staging variant.  Keep
        # non-cluster sweeps focused on code they can actually generate.
        sp_default = mc.EPILOGUE_SPLIT_OPTS if tier.get("cluster") else [0]
        sp_opts = _opts(filters, "split_epilogue", sp_default)
        tma_default = mc.EPILOGUE_TMA_PIPELINED_OPTS if tier.get("persistent_ok") else [0]
        tma_opts = tma_filter if tma_filter is not None else tma_default
        tma_store_stage_opts = (
            tma_store_stage_filter if tma_store_stage_filter is not None
            else mc.TMA_STORE_STAGES_OPTS
        )
        single_default = mc.SINGLE_TMEM_ACCUM_OPTS if tier.get("persistent_ok") else [0]
        single_opts = single_filter if single_filter is not None else single_default
        for bm, bn, bk, ns, gsm, nw, pers, ldw, ov, sp, l1, tma, single_tmem in itertools.product(
            mc.BM_OPTS, bn_list, mc.BK_OPTS, ns_list, gsm_list, nw_list,
            pers_opts, ld_list, ov_opts, sp_opts,
            l1_list, tma_opts, single_opts
        ):
            if single_tmem_policy == "bn512-only":
                if bn == 512 and single_tmem != 1:
                    continue
                if bn != 512 and single_tmem != 0:
                    continue
            stage_opts = tma_store_stage_opts if tma else [2]
            for tma_store_stages in stage_opts:
                yield tier, dict(bm=bm, bn=bn, bk=bk, ns=ns, gsm=gsm, nw=nw,
                                 persistent=pers, ld_width=ldw, overlap=ov,
                                 split_epilogue=sp, l1_no_alloc=l1,
                                 tma_pipelined=tma, tma_store_stages=tma_store_stages,
                                 single_tmem=single_tmem)


def is_valid(tier, k) -> bool:
    """True if `validate_config` raises no warnings for this (tier, knobs)."""
    warnings = mc.validate_config(
        k["bm"], k["bn"], k["bk"], k["ns"], k["gsm"], k["nw"],
        cluster=tier["cluster"],
        persistent=k.get("persistent", 0),
        persistent_ok=tier.get("persistent_ok", False),
        ld_width=k.get("ld_width", 8),
        overlap=k.get("overlap", 0),
        split_epilogue=k.get("split_epilogue", 0),
        l1_no_alloc=k.get("l1_no_alloc", 0),
        tma_pipelined=k.get("tma_pipelined", 0),
        tma_store_stages=k.get("tma_store_stages", 2),
        single_tmem=k.get("single_tmem", 0))
    return not warnings


def valid_combos(tier_dirs, filters=None):
    """Yield only the (tier, knobs) combos that pass `validate_config`."""
    for tier, k in all_combos(tier_dirs, filters):
        if is_valid(tier, k):
            yield tier, k
