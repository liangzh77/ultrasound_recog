import os

from src.research_runtime import (
    ResourcePolicy,
    ResourceSnapshot,
    RuntimeGuard,
    configure_conservative_threads,
    evaluate_runtime,
    evaluate_training_start,
    parse_nvidia_smi_row,
)


def snapshot(**overrides):
    values = {
        "disk_free_gb": 40.0,
        "memory_available_gb": 12.0,
        "cpu_average_percent": 30.0,
        "gpu_memory_used_gb": 1.0,
        "gpu_temperature_c": 60.0,
        "elapsed_hours": 1.0,
    }
    values.update(overrides)
    return ResourceSnapshot(**values)


def test_training_start_rejects_low_memory_and_disk():
    result = evaluate_training_start(
        snapshot(disk_free_gb=19.0, memory_available_gb=7.0),
        ResourcePolicy(),
    )

    assert not result.allowed
    assert result.reasons == (
        "disk_free_below_20gb",
        "memory_available_below_8gb",
    )


def test_runtime_uses_soft_deadline_before_hard_deadline():
    policy = ResourcePolicy()

    assert evaluate_runtime(snapshot(elapsed_hours=11.6), policy).status == (
        "TIME_BUDGET_REACHED"
    )
    assert evaluate_runtime(snapshot(elapsed_hours=23.6), policy).status == (
        "HARD_TIME_LIMIT_REACHED"
    )


def test_runtime_stops_immediately_for_critically_low_memory():
    result = evaluate_runtime(
        snapshot(memory_available_gb=3.5),
        ResourcePolicy(),
    )

    assert result.status == "RESOURCE_GUARD_STOPPED"
    assert result.reasons == ("memory_available_below_4gb",)


def test_runtime_guard_ignores_spike_and_stops_only_after_sustained_pressure():
    guard = RuntimeGuard(ResourcePolicy(), sustained_pressure_seconds=300)

    first = guard.evaluate(
        snapshot(elapsed_hours=1.0, cpu_average_percent=100.0)
    )
    recovered = guard.evaluate(
        snapshot(elapsed_hours=1.03, cpu_average_percent=20.0)
    )
    second_spike = guard.evaluate(
        snapshot(elapsed_hours=2.0, cpu_average_percent=90.0)
    )
    four_minutes = guard.evaluate(
        snapshot(elapsed_hours=2.066, cpu_average_percent=90.0)
    )
    five_minutes = guard.evaluate(
        snapshot(elapsed_hours=2.084, cpu_average_percent=90.0)
    )

    assert first.status == "CONTINUE"
    assert recovered.status == "CONTINUE"
    assert second_spike.status == "CONTINUE"
    assert four_minutes.status == "CONTINUE"
    assert five_minutes.status == "RESOURCE_GUARD_STOPPED"
    assert five_minutes.reasons == ("cpu_average_above_80pct_for_5min",)


def test_runtime_guard_requires_sustained_critical_gpu_temperature():
    guard = RuntimeGuard(ResourcePolicy(), sustained_pressure_seconds=300)

    assert guard.evaluate(
        snapshot(elapsed_hours=0.0, gpu_temperature_c=89.0)
    ).allowed
    result = guard.evaluate(
        snapshot(elapsed_hours=0.1, gpu_temperature_c=89.0)
    )

    assert result.status == "RESOURCE_GUARD_STOPPED"
    assert result.reasons == ("gpu_temperature_above_88c_for_5min",)


def test_conservative_thread_configuration_sets_expected_limits(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    configure_conservative_threads()

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "2"


def test_parses_nvidia_smi_metrics_in_gib_and_celsius():
    result = parse_nvidia_smi_row("NVIDIA GeForce RTX 3080, 10240, 2519, 49, 2")

    assert result["gpu_name"] == "NVIDIA GeForce RTX 3080"
    assert result["gpu_memory_total_gb"] == 10.0
    assert result["gpu_memory_used_gb"] == 2.46
    assert result["gpu_temperature_c"] == 49.0
    assert result["gpu_utilization_percent"] == 2.0
