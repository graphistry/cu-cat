"""
Tests generating.py (synthetic dataset generation).
"""

import numpy as np

<<<<<<<< HEAD:skrub/datasets/tests/test_generating.py
from skrub.datasets._generating import make_deduplication_data
========
from cu_cat.datasets._generating import make_deduplication_data

>>>>>>>> cu-cat/DT5:cu_cat/datasets/tests/test_generating.py


def test_make_deduplication_data():
    np.random.seed(123)
    assert make_deduplication_data(["abc", "cba", "test1"], [3, 2, 1], 0.3) == [
        "agr",
        "abc",
        "abc",
        "cba",
        "cba",
        "test1",
    ]
    assert make_deduplication_data(["abc", "cba", "test1"], [1, 2, 3], 0.8) == [
        "pbc",
        "pza",
        "cba",
        "erxt1",
        "test1",
        "test1",
    ]
