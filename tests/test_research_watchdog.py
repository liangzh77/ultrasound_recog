import subprocess
import sys

from src.research_watchdog import run_with_hard_timeout


def test_watchdog_returns_child_exit_code_without_timeout(tmp_path):
    result = run_with_hard_timeout(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert result.returncode == 7
    assert result.timed_out is False


def test_watchdog_terminates_child_at_hard_timeout(tmp_path):
    result = run_with_hard_timeout(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tmp_path,
        timeout_seconds=0.1,
    )

    assert result.returncode == 124
    assert result.timed_out is True
