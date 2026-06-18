from ai_toolbox.abliteration.hardware import HardwareProfile, detect_hardware


def test_commit_budget_sums_available_ram_and_free_pagefile():
    p = HardwareProfile(available_ram_gb=8.0, pagefile_free_gb=4.0)
    assert p.commit_budget_gb == 12.0


def test_detect_hardware_never_raises_and_returns_profile():
    # Must degrade gracefully even if psutil/torch are missing or fail.
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.total_ram_gb >= 0
