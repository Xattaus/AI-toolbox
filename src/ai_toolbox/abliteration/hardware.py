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


@dataclass
class MemoryEstimate:
    peak_commit_gb: float           # RAM + pagefile need (the 1455 metric)
    peak_vram_gb: float
    breakdown: Dict[str, float] = field(default_factory=dict)


def estimate_cost(model_info: Dict[str, Any], config: Any) -> MemoryEstimate:
    """Estimate peak commit (RAM+pagefile) and VRAM for an abliteration run."""
    params_b = float(model_info.get("estimated_params_b", 0.0) or 0.0)
    hidden = int(model_info.get("hidden_size", 4096) or 4096)
    layers = int(model_info.get("num_layers", 32) or 32)
    batch = int(getattr(config, "batch_size", 8) or 8)

    weights_gb = params_b * 2.0                       # fp16: 2 bytes/param
    activations_gb = (2 * hidden * layers * batch * 512) / _GB
    overhead_gb = 2.0                                 # framework + CUDA context

    breakdown: Dict[str, float] = {
        "weights_fp16": round(weights_gb, 1),
        "activations": round(activations_gb, 2),
        "overhead": overhead_gb,
    }
    if getattr(config, "use_capability_preservation", False):
        breakdown["capability_preservation"] = round(weights_gb * 0.10, 2)
    if getattr(config, "use_direction_selection", False):
        breakdown["direction_selection"] = round(weights_gb * 0.10, 2)

    base = (
        weights_gb + activations_gb + overhead_gb
        + breakdown.get("capability_preservation", 0.0)
        + breakdown.get("direction_selection", 0.0)
    )

    # Auto-tune reloads the model -> a second full copy must be committed.
    auto_tune_mult = 2.0 if getattr(config, "use_auto_tune", False) else 1.0
    breakdown["auto_tune_multiplier"] = auto_tune_mult
    peak_commit = base * auto_tune_mult

    offload = getattr(config, "offload_mode", "auto")
    if offload in ("sequential_cpu", "sequential_disk"):
        # Roughly one layer resident on the GPU at a time.
        peak_vram = (weights_gb / max(layers, 1)) + activations_gb + overhead_gb
    else:
        peak_vram = weights_gb + activations_gb + overhead_gb

    return MemoryEstimate(
        peak_commit_gb=round(peak_commit, 1),
        peak_vram_gb=round(peak_vram, 1),
        breakdown=breakdown,
    )


@dataclass
class RecommendedSettings:
    offload_mode: str
    batch_size: int
    enable_auto_tune: bool
    notes: List[str] = field(default_factory=list)


def recommend_config(
    profile: HardwareProfile,
    model_info: Dict[str, Any],
    conservative: bool = False,
) -> RecommendedSettings:
    """Pick offload mode, batch size and auto-tune based on hardware."""
    params_b = float(model_info.get("estimated_params_b", 0.0) or 0.0)
    weights_gb = params_b * 2.0
    vram_free = profile.vram_free_gb or 0.0
    notes: List[str] = []

    if not profile.cuda_available:
        offload, batch = "sequential_cpu", 1
        notes.append("Ei CUDA-GPU:ta — CPU-offload, batch 1.")
    elif vram_free >= weights_gb * 1.2:
        offload = "auto"
        batch = 2 if conservative else 4
        notes.append(f"VRAM riittää malliin (~{weights_gb:.1f} GB) — auto-offload.")
    else:
        offload = "sequential_cpu"
        batch = 1 if conservative else 2
        notes.append(
            f"VRAM ({vram_free:.1f} GB) ei riitä malliin (~{weights_gb:.1f} GB) "
            f"— sequential_cpu."
        )

    budget = profile.commit_budget_gb
    enable_auto_tune = (not conservative) and budget >= weights_gb * 3.0
    if conservative:
        notes.append("Turvallinen profiili: auto-tune pois, batch minimoitu.")
    elif not enable_auto_tune:
        notes.append("Vähän commit-muistia — auto-tune oletuksena pois.")

    return RecommendedSettings(offload, batch, enable_auto_tune, notes)
