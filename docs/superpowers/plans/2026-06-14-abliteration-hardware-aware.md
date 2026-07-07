# Laitteistotietoinen abliterointi — toteutussuunnitelma

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tunnista käyttäjän laitteisto (RAM/VRAM/pagefile), suosittele abliteroinnin oletukset sen perusteella, ja estä Windowsin `os error 1455` pre-flight-tarkistuksella ennen mallin latausta.

**Architecture:** Uusi puhdas moduuli `abliteration/hardware.py` tarjoaa detection-, estimointi-, suositus- ja pre-flight-funktiot (ei Rich/questionary). CLI-kerros `_full_abliteration_wizard` orkestroi: näyttää laitteistopaneelin, käyttää suosituksia promptien oletuksina, ja ajaa pre-flight-portin ennen ajon käynnistystä tarjoten turvallisen profiilin tai automaattisen pagefile-säädön.

**Tech Stack:** Python 3, `psutil` (RAM/pagefile), `torch.cuda` (VRAM), `subprocess` + PowerShell/WMI (pagefile-säätö), `questionary` + Rich (CLI), `pytest` 8.3.5.

---

## File Structure

- **Create** `src/ai_toolbox/abliteration/hardware.py` — dataclassit (`HardwareProfile`, `MemoryEstimate`, `RecommendedSettings`, `PreflightResult`) + funktiot (`detect_hardware`, `estimate_cost`, `recommend_config`, `check_preflight`, `recommend_pagefile_gb`, `build_set_pagefile_command`, `apply_pagefile_setting`). Yksi vastuu: muistin/laitteiston päättely. Ei käyttäjävuorovaikutusta.
- **Create** `tests/abliteration/test_hardware.py` — yksikkötestit puhtaalle logiikalle (mockattu profiili, ei oikeaa rautaa).
- **Modify** `src/ai_toolbox/abliteration/__init__.py` — vie uudet symbolit (backward-compat -konventio).
- **Modify** `src/ai_toolbox/cli/abliteration_cmd.py` — `_full_abliteration_wizard` (`:727`): laitteistopaneeli + suositusoletukset (`~:767–1000`), pre-flight-portti (`~:1205–1259`).

Kaikki uusi logiikka `hardware.py`:ssä; CLI vain kutsuu ja tulostaa (CLAUDE.md: ei bisneslogiikkaa CLI:hin).

---

## Task 1: HardwareProfile + detect_hardware

**Files:**
- Create: `src/ai_toolbox/abliteration/hardware.py`
- Test: `tests/abliteration/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/abliteration/test_hardware.py
from ai_toolbox.abliteration.hardware import HardwareProfile, detect_hardware


def test_commit_budget_sums_available_ram_and_free_pagefile():
    p = HardwareProfile(available_ram_gb=8.0, pagefile_free_gb=4.0)
    assert p.commit_budget_gb == 12.0


def test_detect_hardware_never_raises_and_returns_profile():
    # Must degrade gracefully even if psutil/torch are missing or fail.
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/abliteration/test_hardware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ai_toolbox.abliteration.hardware'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ai_toolbox/abliteration/hardware.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/abliteration/test_hardware.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai_toolbox/abliteration/hardware.py tests/abliteration/test_hardware.py
git commit -m "$(cat <<'EOF'
feat(abliteration): add HardwareProfile + detect_hardware

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MemoryEstimate + estimate_cost

**Files:**
- Modify: `src/ai_toolbox/abliteration/hardware.py`
- Test: `tests/abliteration/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/abliteration/test_hardware.py
from types import SimpleNamespace
from ai_toolbox.abliteration.hardware import estimate_cost

_INFO = {"estimated_params_b": 8.0, "hidden_size": 4096, "num_layers": 32}


def _cfg(**kw):
    base = dict(
        batch_size=8, offload_mode="auto", use_auto_tune=False,
        use_capability_preservation=False, use_direction_selection=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_weights_term_is_two_bytes_per_param():
    est = estimate_cost(_INFO, _cfg())
    assert est.breakdown["weights_fp16"] == 16.0  # 8B * 2 bytes


def test_auto_tune_roughly_doubles_commit():
    off = estimate_cost(_INFO, _cfg(use_auto_tune=False)).peak_commit_gb
    on = estimate_cost(_INFO, _cfg(use_auto_tune=True)).peak_commit_gb
    assert on == round(2 * off, 1)


def test_capability_preservation_increases_commit():
    base = estimate_cost(_INFO, _cfg()).peak_commit_gb
    more = estimate_cost(_INFO, _cfg(use_capability_preservation=True)).peak_commit_gb
    assert more > base


def test_sequential_offload_lowers_vram_vs_auto():
    auto = estimate_cost(_INFO, _cfg(offload_mode="auto")).peak_vram_gb
    seq = estimate_cost(_INFO, _cfg(offload_mode="sequential_cpu")).peak_vram_gb
    assert seq < auto
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/abliteration/test_hardware.py -k estimate_cost -v`
Expected: FAIL — `ImportError: cannot import name 'estimate_cost'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ai_toolbox/abliteration/hardware.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/abliteration/test_hardware.py -k estimate_cost -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai_toolbox/abliteration/hardware.py tests/abliteration/test_hardware.py
git commit -m "$(cat <<'EOF'
feat(abliteration): add memory cost estimator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: RecommendedSettings + recommend_config

**Files:**
- Modify: `src/ai_toolbox/abliteration/hardware.py`
- Test: `tests/abliteration/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/abliteration/test_hardware.py
from ai_toolbox.abliteration.hardware import recommend_config


def test_no_cuda_recommends_cpu_offload_batch_one():
    p = HardwareProfile(available_ram_gb=32, pagefile_free_gb=32, cuda_available=False)
    rec = recommend_config(p, _INFO)
    assert rec.offload_mode == "sequential_cpu"
    assert rec.batch_size == 1


def test_ample_vram_recommends_auto_offload():
    p = HardwareProfile(available_ram_gb=64, pagefile_free_gb=64,
                        cuda_available=True, vram_free_gb=24.0)
    rec = recommend_config(p, _INFO)  # 8B -> ~16 GB weights, 24 GB free fits
    assert rec.offload_mode == "auto"
    assert rec.batch_size >= 2


def test_tight_vram_recommends_sequential():
    p = HardwareProfile(available_ram_gb=32, pagefile_free_gb=32,
                        cuda_available=True, vram_free_gb=6.0)
    rec = recommend_config(p, _INFO)
    assert rec.offload_mode == "sequential_cpu"


def test_conservative_disables_auto_tune():
    p = HardwareProfile(available_ram_gb=64, pagefile_free_gb=64,
                        cuda_available=True, vram_free_gb=24.0)
    rec = recommend_config(p, _INFO, conservative=True)
    assert rec.enable_auto_tune is False


def test_low_commit_budget_disables_auto_tune():
    p = HardwareProfile(available_ram_gb=8, pagefile_free_gb=8,
                        cuda_available=True, vram_free_gb=24.0)
    rec = recommend_config(p, _INFO)  # budget 16 < 3*16 weights
    assert rec.enable_auto_tune is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/abliteration/test_hardware.py -k recommend_config -v`
Expected: FAIL — `ImportError: cannot import name 'recommend_config'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ai_toolbox/abliteration/hardware.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/abliteration/test_hardware.py -k recommend_config -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai_toolbox/abliteration/hardware.py tests/abliteration/test_hardware.py
git commit -m "$(cat <<'EOF'
feat(abliteration): add hardware-based config recommendation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PreflightResult + check_preflight

**Files:**
- Modify: `src/ai_toolbox/abliteration/hardware.py`
- Test: `tests/abliteration/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/abliteration/test_hardware.py
from ai_toolbox.abliteration.hardware import check_preflight, MemoryEstimate


def _est(commit, vram):
    return MemoryEstimate(peak_commit_gb=commit, peak_vram_gb=vram, breakdown={})


def test_preflight_ok_when_budget_far_exceeds_need():
    p = HardwareProfile(available_ram_gb=64, pagefile_free_gb=64,
                        cuda_available=True, vram_free_gb=24.0)
    r = check_preflight(p, _est(20.0, 18.0), _INFO)
    assert r.status == "ok"
    assert r.bottleneck is None


def test_preflight_fail_on_commit_marks_pagefile_on_windows():
    p = HardwareProfile(available_ram_gb=8, pagefile_free_gb=8,
                        pagefile_total_gb=16, cuda_available=True, vram_free_gb=24.0)
    r = check_preflight(p, _est(40.0, 18.0), _INFO, is_windows=True)
    assert r.status == "fail"
    assert r.bottleneck == "pagefile"
    assert r.shortfall_gb > 0
    assert r.safe_profile is not None
    assert r.recommended_pagefile_gb is not None


def test_preflight_fail_on_vram():
    p = HardwareProfile(available_ram_gb=128, pagefile_free_gb=128,
                        cuda_available=True, vram_free_gb=4.0)
    r = check_preflight(p, _est(20.0, 18.0), _INFO, is_windows=True)
    assert r.status == "fail"
    assert r.bottleneck == "vram"


def test_preflight_warn_when_within_margin():
    p = HardwareProfile(available_ram_gb=10, pagefile_free_gb=11,
                        cuda_available=True, vram_free_gb=24.0)
    # budget 21, need 20 -> headroom 1 < 10% of 20 -> warn
    r = check_preflight(p, _est(20.0, 18.0), _INFO, is_windows=True)
    assert r.status == "warn"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/abliteration/test_hardware.py -k preflight -v`
Expected: FAIL — `ImportError: cannot import name 'check_preflight'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ai_toolbox/abliteration/hardware.py`:

```python
@dataclass
class PreflightResult:
    status: str                                  # "ok" | "warn" | "fail"
    bottleneck: Optional[str]                    # "ram" | "pagefile" | "vram" | None
    shortfall_gb: float
    safe_profile: Optional[RecommendedSettings]
    recommended_pagefile_gb: Optional[int]
    message: str


def check_preflight(
    profile: HardwareProfile,
    estimate: MemoryEstimate,
    model_info: Dict[str, Any],
    margin: float = 0.10,
    is_windows: Optional[bool] = None,
) -> PreflightResult:
    """Compare the estimate against available memory before loading the model."""
    if is_windows is None:
        is_windows = platform.system() == "Windows"

    commit_budget = profile.commit_budget_gb
    commit_need = estimate.peak_commit_gb * (1 + margin)
    commit_short = commit_need - commit_budget

    vram_short = 0.0
    if profile.vram_free_gb is not None:
        vram_short = estimate.peak_vram_gb * (1 + margin) - profile.vram_free_gb

    bottleneck: Optional[str] = None
    shortfall = 0.0
    status = "ok"

    if commit_short > 0 and commit_short >= vram_short:
        # Growing the pagefile is the realistic software fix on Windows;
        # elsewhere the lever is physical RAM.
        bottleneck = "pagefile" if is_windows else "ram"
        shortfall = round(commit_short, 1)
        status = "fail"
    elif vram_short > 0:
        bottleneck = "vram"
        shortfall = round(vram_short, 1)
        status = "fail"
    elif commit_budget - estimate.peak_commit_gb < estimate.peak_commit_gb * margin:
        status = "warn"

    safe_profile: Optional[RecommendedSettings] = None
    recommended_pf: Optional[int] = None
    if status in ("warn", "fail"):
        safe_profile = recommend_config(profile, model_info, conservative=True)
        if bottleneck == "pagefile":
            recommended_pf = recommend_pagefile_gb(estimate)[1]

    message = _preflight_message(status, bottleneck, shortfall)
    return PreflightResult(status, bottleneck, shortfall, safe_profile,
                           recommended_pf, message)


def _preflight_message(status: str, bottleneck: Optional[str], shortfall: float) -> str:
    if status == "ok":
        return "Muisti riittää valitulle konfiguraatiolle."
    labels = {
        "pagefile": "sivutustiedosto (pagefile) liian pieni",
        "ram": "RAM-muisti ei riitä",
        "vram": "GPU-muisti (VRAM) ei riitä",
    }
    label = labels.get(bottleneck or "", "muisti ei riitä")
    verb = "varoitus" if status == "warn" else "ei riitä"
    return f"Pre-flight {verb}: {label} (vajaus ~{shortfall:.1f} GB)."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/abliteration/test_hardware.py -k preflight -v`
Expected: PASS (4 passed). (`recommend_pagefile_gb` is added in Task 5; it is only called on the pagefile branch — the failing test there will pass once Task 5 lands. Run the full file after Task 5.)

> NOTE: `test_preflight_fail_on_commit_marks_pagefile_on_windows` calls `recommend_pagefile_gb`, defined in Task 5. Implement Task 5 immediately after Step 3 here, OR temporarily stub `recommend_pagefile_gb` returning `(0, 0)`. Cleanest: do Task 5 Step 3 now, then run both. To keep commits clean, this task's commit (Step 5) is deferred until Task 5 Step 3 is in place.

- [ ] **Step 5: Commit (after Task 5 Step 3 exists)**

```bash
git add src/ai_toolbox/abliteration/hardware.py tests/abliteration/test_hardware.py
git commit -m "$(cat <<'EOF'
feat(abliteration): add pre-flight memory check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Pagefile recommendation + command builder

**Files:**
- Modify: `src/ai_toolbox/abliteration/hardware.py`
- Test: `tests/abliteration/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/abliteration/test_hardware.py
from ai_toolbox.abliteration.hardware import (
    recommend_pagefile_gb, build_set_pagefile_command,
)


def test_recommend_pagefile_covers_need_and_caps_at_64():
    init, mx = recommend_pagefile_gb(_est(40.0, 18.0))
    assert init >= 40 and mx >= init
    assert mx <= 64  # capped


def test_recommend_pagefile_has_floor():
    init, mx = recommend_pagefile_gb(_est(2.0, 2.0))
    assert init >= 16 and mx >= 32


def test_build_pagefile_command_contains_sizes_and_class():
    cmd = build_set_pagefile_command(32, 48, drive="C:")
    assert "Win32_PageFileSetting" in cmd
    assert "32768" in cmd          # 32 GB in MB
    assert "49152" in cmd          # 48 GB in MB
    assert "AutomaticManagedPagefile" in cmd
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/abliteration/test_hardware.py -k pagefile -v`
Expected: FAIL — `ImportError: cannot import name 'recommend_pagefile_gb'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ai_toolbox/abliteration/hardware.py`:

```python
def recommend_pagefile_gb(
    estimate: MemoryEstimate,
    cap_gb: int = 64,
) -> Tuple[int, int]:
    """Recommend (initial, maximum) pagefile size in GB for the estimate.

    Floors at 16/32 GB, caps both at cap_gb so it never proposes something
    absurd on a huge model.
    """
    need = estimate.peak_commit_gb
    initial = int(min(max(round(need * 1.5), 16), cap_gb))
    maximum = int(min(max(round(need * 2.0), 32), cap_gb))
    maximum = max(maximum, initial)
    return initial, maximum


def build_set_pagefile_command(initial_gb: int, max_gb: int, drive: str = "C:") -> str:
    """Build the elevated PowerShell/WMI command that sets the pagefile.

    Returns the inner PowerShell command string (not executed here).
    """
    init_mb = initial_gb * 1024
    max_mb = max_gb * 1024
    name = f"{drive}\\\\pagefile.sys"
    return (
        "$cs = Get-CimInstance Win32_ComputerSystem; "
        "if ($cs.AutomaticManagedPagefile) { "
        "Set-CimInstance -InputObject $cs "
        "-Property @{AutomaticManagedPagefile=$false} }; "
        f"$pf = Get-CimInstance Win32_PageFileSetting "
        f"-Filter \"Name='{name}'\"; "
        f"if ($pf) {{ Set-CimInstance -InputObject $pf "
        f"-Property @{{InitialSize={init_mb};MaximumSize={max_mb}}} }} "
        f"else {{ New-CimInstance -ClassName Win32_PageFileSetting "
        f"-Property @{{Name='{drive}\\pagefile.sys';"
        f"InitialSize={init_mb};MaximumSize={max_mb}}} }}"
    )


def apply_pagefile_setting(initial_gb: int, max_gb: int, drive: str = "C:") -> bool:
    """Apply the pagefile setting via an elevated (UAC) PowerShell process.

    Windows-only. Returns True on success. Does NOT reboot — the caller must
    tell the user a restart is required. Returns False if not Windows, if the
    user declines the UAC prompt, or on any error (caller shows manual steps).
    """
    if platform.system() != "Windows":
        return False
    inner = build_set_pagefile_command(initial_gb, max_gb, drive)
    try:
        completed = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Start-Process powershell -Verb RunAs -Wait "
                f"-ArgumentList '-NoProfile','-Command',\"{inner}\"",
            ],
            capture_output=True, text=True, timeout=120,
        )
        return completed.returncode == 0
    except Exception:
        return False
```

- [ ] **Step 4: Run full test file to verify all pass**

Run: `python -m pytest tests/abliteration/test_hardware.py -v`
Expected: PASS (all tasks 1–5 tests green, incl. the deferred Task 4 pagefile branch)

- [ ] **Step 5: Commit (covers Task 4 + Task 5 implementations)**

```bash
git add src/ai_toolbox/abliteration/hardware.py tests/abliteration/test_hardware.py
git commit -m "$(cat <<'EOF'
feat(abliteration): add pagefile recommendation and WMI command builder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Export symbols from package __init__

**Files:**
- Modify: `src/ai_toolbox/abliteration/__init__.py`

- [ ] **Step 1: Inspect current exports**

Run: `python -c "import ai_toolbox.abliteration as a; print(a.__file__)"`
Then read the file to match its existing export style (`from .module import X`, `__all__`).

- [ ] **Step 2: Add exports**

Add to `src/ai_toolbox/abliteration/__init__.py` (follow the file's existing pattern; if it has `__all__`, append the names):

```python
from .hardware import (
    HardwareProfile,
    MemoryEstimate,
    RecommendedSettings,
    PreflightResult,
    detect_hardware,
    estimate_cost,
    recommend_config,
    check_preflight,
    recommend_pagefile_gb,
    build_set_pagefile_command,
    apply_pagefile_setting,
)
```

- [ ] **Step 3: Verify import**

Run: `python -c "from ai_toolbox.abliteration import detect_hardware, check_preflight; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/ai_toolbox/abliteration/__init__.py
git commit -m "$(cat <<'EOF'
feat(abliteration): export hardware helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CLI — hardware panel + recommended defaults

**Files:**
- Modify: `src/ai_toolbox/cli/abliteration_cmd.py` (method `_full_abliteration_wizard`, `:727`)

- [ ] **Step 1: Add import at top of file**

Find the existing abliteration imports near the top of `abliteration_cmd.py` and add:

```python
from ..abliteration.hardware import (
    detect_hardware,
    estimate_cost,
    recommend_config,
    check_preflight,
    recommend_pagefile_gb,
    build_set_pagefile_command,
    apply_pagefile_setting,
)
```

- [ ] **Step 2: Insert hardware detection + panel after the model-info block**

In `_full_abliteration_wizard`, immediately AFTER the `if info.get("is_llama31"):` block (around `:767`, right before the `# 2. STRENGTH` section), insert:

```python
        # =====================================================================
        # 1b. HARDWARE DETECTION
        # =====================================================================
        hw = detect_hardware()
        rec = recommend_config(hw, info)

        vram_txt = (
            f"{hw.vram_free_gb:.1f} / {hw.vram_total_gb:.1f} GB ({hw.gpu_name})"
            if hw.cuda_available and hw.vram_total_gb is not None
            else "Ei CUDA-GPU:ta"
        )
        console.print("\n")
        console.print(Panel(
            f"[bold]GPU / VRAM:[/bold]    {vram_txt}\n"
            f"[bold]RAM:[/bold]          {hw.available_ram_gb:.1f} / "
            f"{hw.total_ram_gb:.1f} GB vapaana\n"
            f"[bold]Pagefile:[/bold]     {hw.pagefile_free_gb:.1f} / "
            f"{hw.pagefile_total_gb:.1f} GB vapaana\n"
            f"[bold]Commit-budjetti:[/bold] {hw.commit_budget_gb:.1f} GB\n\n"
            f"[dim]Suositus: offload={rec.offload_mode}, batch={rec.batch_size}, "
            f"auto-tune={'on' if rec.enable_auto_tune else 'off'}[/dim]",
            title="[bold cyan]🖥️  Laitteisto havaittu[/bold cyan]",
            border_style="cyan",
        ))
```

- [ ] **Step 3: Use recommended offload default**

In the OFFLOAD MODE `questionary.select(...)` (`:850`), add the `default` argument so the recommended mode is preselected:

```python
        offload_mode = questionary.select(
            "Offload mode:",
            choices=[
                # ... existing choices unchanged ...
            ],
            default=rec.offload_mode,
            style=custom_style,
            qmark=">>",
            pointer=">"
        ).ask()
```

(questionary matches `default` against each choice's `value`; `rec.offload_mode` is always one of `auto`/`sequential_cpu`/`sequential_disk`.)

- [ ] **Step 4: Use recommended batch default**

In the BATCH SIZE `questionary.text(...)` (`:886`), change the default:

```python
        batch_size_str = questionary.text(
            "Batch size (default 8):",
            default=str(rec.batch_size),
            style=custom_style,
        ).ask()
```

- [ ] **Step 5: Use recommended auto-tune default**

Find the auto-tune confirm prompt (search `"Enable Auto-tuning"` in the file, ~`:990`) and set its `default` to `rec.enable_auto_tune`:

```python
        use_auto_tune = questionary.confirm(
            "Enable Auto-tuning?",
            default=rec.enable_auto_tune,
            style=custom_style,
        ).ask()
```

(Keep the surrounding explanatory `console.print` lines as-is. Only the `default=` changes.)

- [ ] **Step 6: Smoke-test the wizard renders**

Run: `python -c "from ai_toolbox.cli.abliteration_cmd import AbliterationCommands; print('import ok')"`
Expected: `import ok` (no syntax/import error). Full interactive run is manual.

- [ ] **Step 7: Commit**

```bash
git add src/ai_toolbox/cli/abliteration_cmd.py
git commit -m "$(cat <<'EOF'
feat(abliteration): show hardware panel and recommended defaults in wizard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: CLI — pre-flight gate before run

**Files:**
- Modify: `src/ai_toolbox/cli/abliteration_cmd.py` (`_full_abliteration_wizard`, `:1205–1259`)

- [ ] **Step 1: Replace the estimate + summary + confirm block**

Replace the block from `# Estimate requirements` / `reqs = self.abliterator.estimate_requirements(...)` (`:1205–1206`) through the existing `if not questionary.confirm("Aloitetaanko abliteration?", ...): return` (`:1258–1259`) with the pre-flight loop below. The pieces that build `prompt_info`, `extra_targets_str`, `smart_mode_str`, `advanced_str` (`:1208–1236`) stay UNCHANGED — keep them, then replace from the summary `console.print(Panel(...))` onward.

```python
        # =====================================================================
        # PRE-FLIGHT: estimate memory and gate before loading the model
        # =====================================================================
        from types import SimpleNamespace

        while True:
            est_cfg = SimpleNamespace(
                batch_size=batch_size,
                offload_mode=offload_mode,
                use_auto_tune=use_auto_tune,
                use_capability_preservation=use_capability_preservation,
                use_direction_selection=use_direction_selection,
            )
            estimate = estimate_cost(info, est_cfg)
            preflight = check_preflight(hw, estimate, info)

            status_color = {"ok": "green", "warn": "yellow", "fail": "red"}[preflight.status]
            console.print("\n")
            console.print(Panel(
                f"[bold]Source Model:[/bold]   {model_path.name}\n"
                f"[bold]Architecture:[/bold]   {info.get('architecture', 'Unknown')}\n"
                f"[bold]Strength:[/bold]       {strength:.1f}\n"
                f"[bold]Method:[/bold]         {method_choice}\n"
                f"[bold]Smart Mode:[/bold]     {smart_mode_str}\n"
                f"[bold]Advanced:[/bold]       {advanced_str}\n"
                f"[bold]Offload Mode:[/bold]   {offload_mode}\n"
                f"[bold]Batch Size:[/bold]     {batch_size}\n"
                f"[bold]Auto-tune:[/bold]      {'on' if use_auto_tune else 'off'}\n"
                f"[bold]Prompts:[/bold]        {prompt_info}\n"
                f"[bold]Extra Targets:[/bold]  {extra_targets_str}\n"
                f"[bold]Output:[/bold]         {output_name}\n\n"
                f"[dim]Arvioitu commit (RAM+pagefile):[/dim] "
                f"{estimate.peak_commit_gb:.1f} GB / budjetti {hw.commit_budget_gb:.1f} GB\n"
                f"[dim]Arvioitu VRAM:[/dim]   {estimate.peak_vram_gb:.1f} GB\n"
                f"[{status_color}]Pre-flight: {preflight.message}[/{status_color}]",
                title="[bold cyan]Configuration Summary[/bold cyan]",
                border_style=status_color,
            ))

            if preflight.status == "ok":
                if not questionary.confirm(
                    "Aloitetaanko abliteration?", default=True, style=custom_style
                ).ask():
                    return
                break

            # warn/fail -> offer remediation
            choices = []
            if preflight.safe_profile is not None:
                sp = preflight.safe_profile
                choices.append(questionary.Choice(
                    title=(
                        f"Käytä turvallista profiilia "
                        f"(offload={sp.offload_mode}, batch={sp.batch_size}, "
                        f"auto-tune={'on' if sp.enable_auto_tune else 'off'})"
                    ),
                    value="safe",
                ))
            if preflight.bottleneck == "pagefile" and preflight.recommended_pagefile_gb:
                choices.append(questionary.Choice(
                    title=(
                        f"Säädä pagefile automaattisesti "
                        f"(~{preflight.recommended_pagefile_gb} GB, vaatii admin + reboot)"
                    ),
                    value="pagefile",
                ))
            choices.append(questionary.Choice(title="Jatka silti (oma vastuu)", value="continue"))
            choices.append(questionary.Choice(title="Peruuta", value="cancel"))

            console.print(f"\n[{status_color}]⚠️  {preflight.message}[/{status_color}]")
            action = questionary.select(
                "Miten jatketaan?",
                choices=choices,
                style=custom_style,
                qmark=">>",
                pointer=">",
            ).ask()

            if action == "safe":
                sp = preflight.safe_profile
                batch_size = sp.batch_size
                offload_mode = sp.offload_mode
                use_auto_tune = sp.enable_auto_tune
                console.print("[green]Turvallinen profiili otettu käyttöön.[/green]")
                continue  # re-estimate and re-show summary
            elif action == "pagefile":
                init_gb, max_gb = recommend_pagefile_gb(estimate)
                if questionary.confirm(
                    f"Asetetaanko pagefile {init_gb}–{max_gb} GB? "
                    f"(UAC-kehote avautuu, käynnistä kone uudelleen jälkeenpäin)",
                    default=True, style=custom_style,
                ).ask():
                    if apply_pagefile_setting(init_gb, max_gb):
                        console.print(
                            "[green]Pagefile asetettu.[/green] "
                            "[yellow]Käynnistä kone uudelleen ja aja abliterointi "
                            "uudelleen.[/yellow]"
                        )
                        return
                    else:
                        console.print(
                            "[yellow]Automaattinen säätö ei onnistunut "
                            "(ei admin-oikeuksia / peruutettu).[/yellow]\n"
                            f"[dim]Aseta käsin: Win+R → SystemPropertiesAdvanced → "
                            f"Suorituskyky → Asetukset → Lisäasetukset → Näennäismuisti → "
                            f"alkukoko {init_gb*1024} MB, maksimi {max_gb*1024} MB → reboot.[/dim]"
                        )
                continue
            elif action == "continue":
                console.print("[yellow]Jatketaan varoituksesta huolimatta.[/yellow]")
                break
            else:  # cancel / None
                return
```

- [ ] **Step 2: Verify the file still imports**

Run: `python -c "from ai_toolbox.cli.abliteration_cmd import AbliterationCommands; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Run the full test suite to confirm nothing broke**

Run: `python -m pytest tests/ -v`
Expected: all hardware tests PASS; no import errors.

- [ ] **Step 4: Commit**

```bash
git add src/ai_toolbox/cli/abliteration_cmd.py
git commit -m "$(cat <<'EOF'
feat(abliteration): pre-flight memory gate prevents os error 1455

Warn/fail before model load; offer safe profile or automatic pagefile
adjustment. Replaces the rough estimate_requirements summary with a real
commit/VRAM estimate vs detected hardware budget.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Manual verification (real run)

**Files:** none (manual)

- [ ] **Step 1: Launch the wizard**

Run: `python -m ai_toolbox` → Training Center → Remove Censorship → Full Abliteration.

- [ ] **Step 2: Verify hardware panel** shows real GPU/VRAM/RAM/pagefile numbers and a recommendation line.

- [ ] **Step 3: Reproduce the original 1455 config** — pick the 8.8B model, auto-tune ON, capability preservation ON, batch 8. Confirm the summary shows **Pre-flight: fail** with bottleneck = pagefile BEFORE any model loading starts.

- [ ] **Step 4: Choose "Käytä turvallista profiilia"** and confirm the summary re-renders with status **ok** (lower batch, auto-tune off, sequential offload).

- [ ] **Step 5 (optional):** Choose the pagefile option, confirm the UAC prompt appears; cancel it and confirm the manual instructions are printed (no crash).

- [ ] **Step 6:** Confirm an OK config still runs the abliteration normally as before.

---

## Self-Review Notes

- **Spec coverage:** detection panel (Task 7), recommended defaults (Task 7), `estimate_cost`/`recommend_config`/`check_preflight` (Tasks 2–4), pagefile auto-adjust + manual fallback (Task 5 + Task 8), warn/fail + safe profile + override (Task 8), graceful degradation (Task 1 `detect_hardware` try/except; CLI uses whatever fields exist), tests (Tasks 1–5), abliteration-only scope (all code in `abliteration/` + its CLI). ✓
- **Cross-task type consistency:** `HardwareProfile`, `MemoryEstimate`, `RecommendedSettings` (`.offload_mode`/`.batch_size`/`.enable_auto_tune`), `PreflightResult` (`.status`/`.bottleneck`/`.safe_profile`/`.recommended_pagefile_gb`/`.message`) used identically in tests and CLI. `recommend_pagefile_gb` returns `(initial, maximum)`; CLI unpacks both; `check_preflight` uses `[1]` (maximum). ✓
- **Ordering caveat:** Task 4's pagefile-branch test depends on Task 5's `recommend_pagefile_gb`; the plan defers Task 4's commit until Task 5 Step 3 exists (noted inline). ✓
