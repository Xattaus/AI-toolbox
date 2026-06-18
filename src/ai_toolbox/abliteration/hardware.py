"""Hardware detection and memory pre-flight for abliteration.

Pure helpers: detect RAM/VRAM/pagefile, estimate the memory an abliteration
run needs, recommend settings, and run a pre-flight check that prevents the
Windows ERROR_COMMITMENT_LIMIT (os error 1455) before the model is loaded.

No Rich/questionary here — the CLI layer owns user interaction. Every probe is
wrapped so detection NEVER raises; missing data degrades to zero/None.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_GB = 1024 ** 3


@dataclass
class HardwareProfile:
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    pagefile_total_gb: float = 0.0
    pagefile_free_gb: float = 0.0
    cuda_available: bool = False
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None

    @property
    def commit_budget_gb(self) -> float:
        """Commit budget = free RAM + free pagefile.

        This is the quantity Windows checks for ERROR_COMMITMENT_LIMIT (1455).
        """
        return round(self.available_ram_gb + self.pagefile_free_gb, 1)


def detect_hardware() -> HardwareProfile:
    """Probe RAM, pagefile and GPU. Never raises; partial data on failure."""
    profile = HardwareProfile()
    try:
        import psutil

        vm = psutil.virtual_memory()
        profile.total_ram_gb = round(vm.total / _GB, 1)
        profile.available_ram_gb = round(vm.available / _GB, 1)
        sw = psutil.swap_memory()
        profile.pagefile_total_gb = round(sw.total / _GB, 1)
        profile.pagefile_free_gb = round(sw.free / _GB, 1)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            profile.cuda_available = True
            profile.gpu_name = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            profile.vram_total_gb = round(total / _GB, 1)
            profile.vram_free_gb = round(free / _GB, 1)
    except Exception:
        pass
    return profile
