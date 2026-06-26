import numpy as np

from mmu.ids import NSIDE, ORDER, assign_global_id


def test_order_and_nside():
    assert ORDER == 29
    assert NSIDE == 2 ** 29


def test_deterministic_int64_and_shape():
    ra = np.array([10.684, 83.822, 201.365])
    dec = np.array([41.269, -5.391, -47.488])
    a = assign_global_id(ra, dec)
    b = assign_global_id(ra, dec)
    assert a.dtype == np.int64
    assert a.shape == ra.shape
    np.testing.assert_array_equal(a, b)


def test_distinct_points_distinct_ids():
    a = assign_global_id(np.array([10.0]), np.array([20.0]))
    b = assign_global_id(np.array([200.0]), np.array([-30.0]))
    assert a[0] != b[0]


def test_scalar_inputs_supported():
    out = assign_global_id(10.684, 41.269)
    assert out.shape == (1,)
    assert out.dtype == np.int64
