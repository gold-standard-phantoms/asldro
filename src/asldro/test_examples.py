""" Example tests - only run when --runslow is passed to pytest """

import pytest
from asldro.examples import run_full_pipeline


@pytest.mark.slow
def test_run_full_pipeline():
    """ Runs the full ASL DRO pipeline """
    run_full_pipeline()
