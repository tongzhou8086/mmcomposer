"""Live leaderboard rendering --- the `leaderboard` module from DESIGN.md.

Pure terminal presentation.  It takes a list of result records (e.g. straight
from ``cache.top_n``) and renders a ranked table; `LiveDisplay` redraws it in
place on a TTY.  It has **no idea** how the results were produced -- no GPU, no
mvp_core, no sweep state.

A *record* is a dict with ``tflops`` (and optional ``vs_cublas``) plus a
``config`` dict holding the knobs (``bn``, ``ns``, ``gsm``, ``nw``,
``persistent``, ``overlap``, ``split_epilogue``, ``l1_no_alloc``,
``tma_pipelined``, ``tma_store_stages``, ``single_tmem``) and the display flags
``ws`` / ``cluster``.  Knobs may also be flat on the record; `_cfg` handles both.

Public API:
    progress_bar(done, total, width=36) -> str
    render(results, shape, *, cublas_tflops, n_combos, top, title, ...) -> str
    LiveDisplay(redraw="auto").update(block)
"""
from __future__ import annotations

import sys


def _cfg(r: dict) -> dict:
    return r.get("config", r)


def progress_bar(done: int, total, width: int = 36) -> str:
    if total and total > 0:
        done = min(done, total)
        frac = done / total
        filled = min(width, int(round(frac * width)))
        return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({frac * 100:5.1f}%)"
    return f"[{'?' * width}] {done} measured"


def _progress_lines(done, total, progress) -> list:
    if done is None:
        return []
    p = progress or {}
    phase = p.get("phase")
    msg = p.get("message") or phase
    if phase and phase not in {"benchmarking", "collecting", "done"}:
        lines = [progress_bar(int(p.get("done") or 0), p.get("total"))]
        lines.append(f"phase: {msg}")
        if total:
            lines.append(f"measured combos: {done}/{total}")
        return lines
    lines = [progress_bar(done, total)]
    if msg and phase in {"benchmarking", "collecting", "done"}:
        lines.append(f"phase: {msg}")
    return lines


def render(results, shape, *, cublas_tflops=None, n_combos=None, top: int = 10,
           title: str = "leaderboard", done=None, total=None, progress=None) -> str:
    """Render a ranked top-`top` table for shape `(M, N, K)`.  `results` is a list
    of records; they're sorted by tflops (desc) defensively."""
    M, N, K = shape
    rows = sorted(results, key=lambda r: (r.get("tflops") if r.get("tflops") is not None
                                          else -1.0), reverse=True)[:top]
    lines = [title]
    lines.extend(_progress_lines(done, total, progress))
    lines.append(f"cuBLAS reference: {cublas_tflops:.0f} TFLOPS" if cublas_tflops
                 else "cuBLAS reference: n/a")
    shown = n_combos if n_combos is not None else len(results)
    lines.append(f"Top {len(rows)} of {shown} measured combos at {M}x{N}x{K}, by TFLOPS:")
    lines.append("")
    hdr = (f"{'#':>2}  {'TFLOPS':>7}  {'%cuBLAS':>7}  {'WS':>3} {'2CTA':>4}  "
           f"{'BN':>3} {'NS':>2} {'GSM':>3} {'NW':>2}  {'PERS':>4} {'OV':>2} "
           f"{'SP':>2} {'L1':>2} {'TMA':>3} {'TMS':>3} {'STM':>3}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(rows, 1):
        c = _cfg(r)
        ws = "on" if c.get("ws", True) else "off"
        cta = "on" if c.get("cluster", c.get("two_cta")) else "off"
        vs = r.get("vs_cublas")
        if vs is None and cublas_tflops and r.get("tflops"):
            vs = r["tflops"] / cublas_tflops
        vsc = f"{vs * 100:.0f}%" if vs else "-"
        lines.append(
            f"{i:>2}  {r.get('tflops', 0):>7.0f}  {vsc:>7}  {ws:>3} {cta:>4}  "
            f"{c.get('bn', 0):>3} {c.get('ns', 0):>2} {c.get('gsm', 0):>3} "
            f"{c.get('nw', 0):>2}  {c.get('persistent', 0):>4} {c.get('overlap', 0):>2} "
            f"{c.get('split_epilogue', 0):>2} {c.get('l1_no_alloc', 0):>2} "
            f"{c.get('tma_pipelined', 0):>3} {c.get('tma_store_stages', 2):>3} "
            f"{c.get('single_tmem', 0):>3}")
    return "\n".join(lines) + "\n"


class LiveDisplay:
    """Redraws a multi-line block in place on a TTY (ANSI cursor-up + clear)."""

    def __init__(self, redraw: str = "auto", stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.redraw = (redraw == "always"
                       or (redraw == "auto" and hasattr(self.stream, "isatty")
                           and self.stream.isatty()))
        self._lines = 0

    def update(self, block: str) -> None:
        if self.redraw and self._lines:
            self.stream.write(f"\x1b[{self._lines}F\x1b[J")
        self.stream.write(block)
        self.stream.flush()
        self._lines = len(block.splitlines())
