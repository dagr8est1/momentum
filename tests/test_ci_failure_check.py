"""Temporary test to verify the CI workflow correctly fails the build.

Delete this file once the CI failure path has been confirmed on GitHub.
"""

import pytest


@pytest.mark.skip(reason="Intentional failure already confirmed CI reports failed runs")
def test_ci_should_fail():
    assert False, "Intentional failure to verify CI reports failed runs"
