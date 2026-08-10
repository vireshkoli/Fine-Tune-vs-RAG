"""Device exclusivity checks.

Added after a real incident rather than on principle. A parity index build was
sharing GPU 0 with an unrelated training job holding 35GB; embedding ran roughly
three times slower, and had a *timed* arm run under those conditions its p50 and
p95 would have been silently wrong — indistinguishable from a genuinely slower
arm. Contention does not raise an error, it just quietly changes your numbers.

So exclusivity is verified and recorded in every result rather than assumed. The
plan called this risk "eliminated" because both GPUs happened to be idle when it
was written; an empty GPU is a snapshot, not a property.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

#: A few hundred MiB of context from our own process is normal; a neighbouring
#: job is orders of magnitude larger.
FOREIGN_MEMORY_TOLERANCE_MIB = 512


@dataclass(frozen=True)
class DeviceOccupancy:
    """Who is using a GPU besides us."""

    device: int
    total_used_mib: int
    foreign_pids: tuple[int, ...]
    foreign_mib: int
    available: bool

    @property
    def is_exclusive(self) -> bool:
        return self.foreign_mib <= FOREIGN_MEMORY_TOLERANCE_MIB

    def as_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "total_used_mib": self.total_used_mib,
            "foreign_pids": list(self.foreign_pids),
            "foreign_mib": self.foreign_mib,
            "exclusive": self.is_exclusive,
        }

    def __str__(self) -> str:
        if not self.available:
            return f"GPU {self.device}: unavailable"
        if self.is_exclusive:
            return f"GPU {self.device}: exclusive ({self.total_used_mib} MiB in use)"
        return (
            f"GPU {self.device}: SHARED — {self.foreign_mib} MiB held by "
            f"pid(s) {', '.join(map(str, self.foreign_pids))}"
        )


def _nvidia_smi(args: list[str]) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def device_occupancy(device: int) -> DeviceOccupancy:
    """Report other processes' memory on ``device``.

    Processes are attributed to devices by **GPU UUID**, not by elimination.
    A first version guessed — it treated any foreign process whose memory fitted
    within the target device's usage as resident there — and promptly reported a
    35GB job on GPU 0 as contending on GPU 1, falsely invalidating a clean run.
    ``nvidia-smi`` will report ``gpu_uuid`` per compute app, so ask for it.
    """
    used = _nvidia_smi(
        [f"--id={device}", "--query-gpu=memory.used,uuid", "--format=csv,noheader,nounits"]
    )
    if not used:
        return DeviceOccupancy(device, 0, (), 0, available=False)

    fields = [f.strip() for f in used[0].split(",")]
    if len(fields) < 2:
        return DeviceOccupancy(device, 0, (), 0, available=False)
    total_used, uuid = int(fields[0]), fields[1]

    ours = os.getpid()
    foreign: list[tuple[int, int]] = []
    for line in _nvidia_smi(
        ["--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"]
    ):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[2] != uuid:
            continue  # a different device
        try:
            pid, mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid != ours:
            foreign.append((pid, mib))

    return DeviceOccupancy(
        device=device,
        total_used_mib=total_used,
        foreign_pids=tuple(pid for pid, _ in foreign),
        foreign_mib=sum(mib for _, mib in foreign),
        available=True,
    )


def assert_device_exclusive(device: int, *, strict: bool = False) -> DeviceOccupancy:
    """Check that ``device`` is ours alone before timing anything.

    ``strict`` raises; otherwise the caller gets the occupancy back and is
    expected to record it in the result JSON, so a reader can see that a run was
    contended instead of wondering why p95 looks odd.
    """
    occupancy = device_occupancy(device)
    if strict and occupancy.available and not occupancy.is_exclusive:
        raise RuntimeError(
            f"{occupancy}. Latency measured under contention is not comparable; "
            f"wait for the device or point inference_device at an idle GPU."
        )
    return occupancy
