import collections
from typing import Any, Hashable
from inspect import getmodule
import numpy as np
from sklearn.utils import check_array
import cupy as cp

try:
    # Works for sklearn >= 1.0
    from sklearn.utils import parse_version  # noqa
except ImportError:
    # Works for sklearn < 1.0
    from sklearn.utils.fixes import _parse_version as parse_version  # noqa


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def check_input(X) -> np.ndarray:
    """
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much
    if 'numpy' in str(getmodule(X)):
        X_ = check_array(
            X,
            dtype=None,
            ensure_2d=True,
            force_all_finite=False,
        )
        # If the array contains both NaNs and strings, convert to object type
        if X_.dtype.kind in {"U", "S"}:  # contains strings
            if np.any(X_ == "nan"):  # missing value converted to string
                return check_array(
                    cp.array(X, dtype=object),
                    dtype=None,
                    ensure_2d=True,
                    force_all_finite=False,
                )
    if 'numpy' not in str(getmodule(X)):
        for k in range(X.shape[1]):
            try:
                X.iloc[:,k]=cudf.to_numeric(X.iloc[:,k], downcast='float').to_cupy()
                print('passing from cudf to cupy')
            except:
                pass
        X_ = X
    
    return X_

def df_type(df):
    """
    Returns df type
    """

    # try: 
    X = str(getmodule(df))
    # except:
    if X == 'None':
    # try:
        X = str(cp.get_array_module(df))
    # except:
    return X
