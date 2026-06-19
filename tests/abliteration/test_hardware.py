from types import SimpleNamespace

from ai_toolbox.abliteration.hardware import HardwareProfile, detect_hardware
from ai_toolbox.abliteration.hardware import estimate_cost

_INFO = {"estimated_params_b": 8.0, "hidden_size": 4096, "num_layers": 32}


def _cfg(**kw):
    base = dict(
        batch_size=8, offload_mode="auto", use_auto_tune=False,
        use_capability_preservation=False, use_direction_selection=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_commit_budget_sums_available_ram_and_free_pagefile():
    p = HardwareProfile(available_ram_gb=8.0, pagefile_free_gb=4.0)
    assert p.commit_budget_gb == 12.0


def test_detect_hardware_never_raises_and_returns_profile():
    # Must degrade gracefully even if psutil/torch are missing or fail.
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb >= 0


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
