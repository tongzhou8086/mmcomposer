"""Kernel compilation --- the `compile` module from DESIGN.md.

Pure source->cubin via ``nvcc --cubin``.  No GPU needed (nvcc targets the arch),
no kernel loading, no launch.  Cubins are cached on disk next to the .cu and only
rebuilt when the source is newer; writes are atomic (temp + rename) so parallel
compiles of the same file can't corrupt each other.

Public API:
    compile_one(src_path, arch="sm_100a")  -> cubin_path        (raises CompileError)
    compile_many(src_paths, arch="sm_100a") -> {src: CompileResult}
    cubin_path_for(src_path, arch="sm_100a") -> str

Note: the standalone download host still uses its own inlined `_runtime.compile_kernel`;
the two are unified when codegen/runtime are promoted (see DESIGN.md).
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

DEFAULT_ARCH = "sm_100a"


class CompileError(RuntimeError):
    """nvcc failed for a source file; carries the captured stderr."""


@dataclass
class CompileResult:
    src: str
    ok: bool
    cubin: str | None = None
    error: str | None = None


def cubin_path_for(src_path: str, arch: str = DEFAULT_ARCH) -> str:
    """Where the cubin for `src_path` lives (next to the .cu, suffixed by arch)."""
    assert src_path.endswith(".cu"), f"expected a .cu file, got {src_path}"
    return src_path[:-3] + f"_{arch}.cubin"


def _up_to_date(src_path: str, cubin_path: str) -> bool:
    return (os.path.exists(cubin_path)
            and os.path.getmtime(cubin_path) >= os.path.getmtime(src_path))


def compile_one(src_path: str, arch: str = DEFAULT_ARCH,
                extra_opts: list | None = None, force: bool = False) -> str:
    """Compile one .cu to a cubin and return its path.

    Skips the build when an up-to-date cubin already exists (unless `force`).
    Raises CompileError on nvcc failure.
    """
    cubin_path = cubin_path_for(src_path, arch)
    if not force and _up_to_date(src_path, cubin_path):
        return cubin_path
    nvcc = os.environ.get("NVCC", "nvcc")
    tmp = f"{cubin_path}.tmp.{os.getpid()}"
    cmd = [nvcc, f"-arch={arch}", "-O3", "--std=c++17", "--cubin",
           *(extra_opts or []), src_path, "-o", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass
        raise CompileError(f"nvcc failed (exit {r.returncode}) for {src_path}:\n{r.stderr}")
    os.replace(tmp, cubin_path)   # atomic publish
    return cubin_path


def compile_many(src_paths, arch: str = DEFAULT_ARCH, workers: int | None = None,
                 extra_opts: list | None = None, force: bool = False) -> dict:
    """Compile many .cu files in parallel (nvcc is a subprocess, so threads are
    enough).  Returns {src_path: CompileResult} -- never raises; per-file failures
    are reported as ok=False so the caller can prune them before any GPU time."""
    src_paths = list(src_paths)
    out: dict[str, CompileResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compile_one, s, arch, extra_opts, force): s for s in src_paths}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                out[s] = CompileResult(src=s, ok=True, cubin=fut.result())
            except CompileError as e:
                out[s] = CompileResult(src=s, ok=False, error=str(e))
    return out
