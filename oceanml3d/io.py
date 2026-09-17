"""Producing files safely when several processes run the same script.

`prepare-obs` writes NetCDF files that every rank then reads. Lightning's DDP relaunches the whole
script once per rank, and `srun` starts them all at once, so without care four processes race to
write the same file. Being idempotent -- "skip if the output exists" -- does not help and in fact
makes it worse: the existence test passes the moment the first writer creates the file, long before
it has finished filling it, so the other ranks sail on and read a truncated NetCDF.

Two pieces, and both are needed: one writer (:func:`is_global_zero`), and a file that only becomes
visible once it is complete (:func:`write_netcdf_atomic`).
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Every launcher's idea of "which process am I". torchrun sets RANK; Slurm sets SLURM_PROCID; MPI
# runners set their own; Lightning's subprocess launcher sets LOCAL_RANK and NODE_RANK. A process is
# globally first only if every one of these that is set says zero -- LOCAL_RANK=0 on node 3 is not.
_RANK_VARS = ("RANK", "SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "NODE_RANK", "LOCAL_RANK")


def global_rank_env() -> dict[str, str]:
    """The rank variables actually set, for error messages."""
    return {v: os.environ[v] for v in _RANK_VARS if os.environ.get(v) not in (None, "")}


def is_global_zero() -> bool:
    """True for a single process, and for exactly one process of a distributed launch."""
    return all(value == "0" for value in global_rank_env().values())


def write_netcdf_atomic(ds: Any, path: str | os.PathLike, **kwargs: Any) -> Path:
    """Write ``ds`` so the destination never exists in a half-written state.

    The temporary name carries the pid, so two writers cannot collide on it either, and
    ``os.replace`` is atomic within a filesystem. A reader waiting on ``path`` therefore sees either
    nothing or a complete file -- which is what makes :func:`wait_for` sound.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        ds.to_netcdf(tmp, **kwargs)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def wait_for(paths: Iterable[str | os.PathLike], timeout: float = 3600.0,
             poll: float = 2.0, what: str = "inputs") -> None:
    """Block until every path exists, or explain what is missing and why we were waiting.

    Used by the non-zero ranks while rank 0 produces the files. Sound only because the producer
    writes atomically; polling for a name that appears before its contents do is the classic way to
    turn a race into a corrupt read rather than a crash.
    """
    pending = [Path(p) for p in paths]
    if not pending:
        return
    deadline = time.monotonic() + timeout
    while True:
        missing = [p for p in pending if not p.exists()]
        if not missing:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"waited {timeout:.0f}s for {what} that rank 0 was expected to produce, still "
                f"missing: {[str(p) for p in missing]}. If rank 0 failed, its traceback is in its "
                f"own log; if the paths differ between ranks, check that paths.root resolves to "
                f"shared storage."
            )
        time.sleep(poll)
