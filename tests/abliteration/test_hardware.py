from types import SimpleNamespace

from ai_toolbox.abliteration.hardware import HardwareProfile, detect_hardware
from ai_toolbox.abliteration.hardware import estimate_cost
from ai_toolbox.abliteration.hardware import recommend_config
from ai_toolbox.abliteration.hardware import check_preflight, MemoryEstimate
from ai_toolbox.abliteration.hardware import (
    recommend_pagefile_gb, build_set_pagefile_command,
)

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
