import numpy as np
from bikeprice.psi import psi, most_drifted

def test_psi_same_distribution_is_near_zero():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) < 1e-6

def test_psi_shift_is_large_and_most_drifted():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 500)
    assert psi(a, a + 3) > 0.25
    assert most_drifted({"a": 0.01, "b": 0.9}) == "b"