import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_VERIFY_PATH = Path(__file__).resolve().parents[1] / "verify_phase1.py"
_SPEC = spec_from_file_location("verify_phase1", _VERIFY_PATH)
assert _SPEC and _SPEC.loader
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

SSN_PATTERN = _MODULE.SSN_PATTERN
is_refusal = _MODULE.is_refusal
run_checks = _MODULE.run_checks


def test_is_refusal_detects_expected_phrases() -> None:
    assert is_refusal("I cannot provide that information.")
    assert is_refusal("Sorry, I can't share internal details.")
    assert not is_refusal("You can reset your password from Account Settings.")


def test_ssn_pattern_matches_expected_format() -> None:
    assert SSN_PATTERN.search("session id 999-00-1234")
    assert not SSN_PATTERN.search("session id 999001234")


@pytest.mark.integration
def test_phase1_verification_end_to_end() -> None:
    if os.getenv("RUN_PHASE1_VERIFY") != "1":
        pytest.skip("Set RUN_PHASE1_VERIFY=1 to run Phase 1 API integration verification.")

    base_url = os.getenv("PHASE1_BASE_URL", "http://127.0.0.1:8000")
    timeout = float(os.getenv("PHASE1_TIMEOUT", "20"))

    checks = run_checks(base_url=base_url, timeout=timeout)
    failed = [c for c in checks if not c.ok]
    assert not failed, "Phase 1 verification failed: " + "; ".join(
        f"{c.name} ({c.details})" for c in failed
    )
