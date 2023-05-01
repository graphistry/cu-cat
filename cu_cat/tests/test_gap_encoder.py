import numpy as np
import cupy as cp
import cudf
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import NotFittedError
from time import time

from cu_cat import GapEncoder
from cu_cat._utils import parse_version
from cu_cat.tests.utils import generate_cudata, generate_cudata


# def test_analyzer():
#     """
#     Test if the output is different when the analyzer is 'word' or 'char'.
#     If it is, no error ir raised.
#     """
#     add_words = False
#     n_samples = 70
#     X = cp.array_str(generate_cudata(n_samples, random_state=0))
#     n_components = 10
#     # Test first analyzer output:
#     encoder = GapEncoder(
#         n_components=n_components,
#         init="k-means++",
#         analyzer="char",
#         add_words=add_words,
#         random_state=42,
#         rescale_W=True,
#     )
#     encoder.fit(X)
#     y = encoder.transform(X)

#     # Test the other analyzer output:
#     encoder = GapEncoder(
#         n_components=n_components,
#         init="k-means++",
#         analyzer="word",
#         add_words=add_words,
#         random_state=42,
#     )
#     encoder.fit(X)
#     y2 = encoder.transform(X)

#     # Test inequality between the word and char analyzers output:
#     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y, y2)


@pytest.mark.parametrize(
    "hashing, init, analyzer, add_words",
    [
        # (False, "k-means++", "word", True),
        (True, "random", "char", False),
        # (True, "k-means", "char_wb", True),
    ],
)
def test_gap_encoder(
    hashing: bool, init: str, analyzer: str, add_words: bool, n_samples: int = 70
) -> None:
    X = generate_cudata(n_samples, random_state=0)

    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        # n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (n_samples, n_components * X.shape[1]), str(y.shape)

    # Test L1-norm of topics W.
    for col_enc in encoder.fitted_models_:
        l1_norm_W = np.abs(col_enc.W_).sum(axis=1)
        np.testing.assert_array_almost_equal(l1_norm_W.get(), np.ones(n_components))

    # Test same seed return the same output
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
        random_state=42,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    np.testing.assert_array_equal(y, y2)


def test_input_type() -> None:
    # Numpy array with one column
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_array = enc.fit_transform(X)
    # List
    X2 = [["alice"], ["bob"]]
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_list = enc.fit_transform(X2)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_list)

    # Numpy array with two columns
    X = np.array([["alice", "charlie"], ["bob", "delta"]])
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_array = enc.fit_transform(X)
    # Pandas dataframe with two columns
    df = pd.DataFrame(X)
    enc = GapEncoder(n_components=2, random_state=42)
    X_enc_df = enc.fit_transform(df)
    # Check if the encoded vectors are the same
    np.testing.assert_array_equal(X_enc_array, X_enc_df)


# def test_partial_fit(n_samples=70) -> None:
#     X = generate_cudata(n_samples, random_state=0)
#     # Gap encoder with fit on one batch
#     enc = GapEncoder(random_state=42, batch_size=n_samples, max_iter=1)
#     X_enc = enc.fit_transform(X)
#     # Gap encoder with partial fit
#     enc = GapEncoder(random_state=42)
#     enc.partial_fit(X)
#     X_enc_partial = enc.transform(X)
#     # Check if the encoded vectors are the same
#     np.testing.assert_almost_equal(X_enc, X_enc_partial)


def test_get_feature_names_out(n_samples=70) -> None:
    X = generate_cudata(n_samples, random_state=0)
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    # Expect a warning if sklearn >= 1.0
    if parse_version(sklearn_version) < parse_version("1.0"):
        feature_names_1 = enc.get_feature_names()
    else:
        with pytest.warns(DeprecationWarning):
            feature_names_1 = enc.get_feature_names()
    feature_names_2 = enc.get_feature_names_out()
    for topic_labels in [feature_names_1, feature_names_2]:
        # Check number of labels
        assert len(topic_labels) == enc.n_components * X.shape[1]
        # Test different parameters for col_names
        topic_labels_2 = enc.get_feature_names_out(col_names="auto")
        assert topic_labels_2[0] == "col0: " + topic_labels[0]
        topic_labels_3 = enc.get_feature_names_out(col_names=["abc", "def"])
        assert topic_labels_3[0] == "abc: " + topic_labels[0]
    return


def test_overflow_error() -> None:
    np.seterr(over="raise", divide="raise")
    r = cp.random.RandomState(0)
    X = r.randint(1e5, 1e6, size=(8000, 1)).astype(str)
    enc = GapEncoder(
        n_components=2, batch_size=1, min_iter=1, max_iter=1, random_state=0
    )
    enc.fit(X)


# def test_score(n_samples: int = 70) -> None:
#     X1 = generate_cudata(n_samples, random_state=0)
#     X2 = np.hstack([X1, X1])
#     enc = GapEncoder(random_state=42)
#     enc.fit(X1)
#     score_X1 = enc.score(X1)
#     enc.fit(X2)
#     score_X2 = enc.score(X2)
#     # Check that two identical columns give the same score
#     assert score_X1 * 2 == score_X2


@pytest.mark.parametrize("missing", ["zero_impute", "error", "aaa"])
def test_missing_values(missing: str) -> None:
    observations = [
        ["alice", "bob"],
        ["bob", "alice"],
        ["bob", np.nan],
        ["alice", "charlie"],
        [np.nan, "alice"],
    ]
    observations = np.array(observations, dtype=object)
    enc = GapEncoder(handle_missing=missing, n_components=3)
    if missing == "error":
        with pytest.raises(ValueError, match="Input data contains missing values"):
            enc.fit_transform(observations)
    elif missing == "zero_impute":
        enc.fit_transform(observations)
        enc.fit(observations) ##not partial_fit
    else:
        with pytest.raises(
            ValueError,
            match=r"handle_missing should be either "
            r"'error' or 'zero_impute', got 'aaa'",
        ):
            enc.fit_transform(observations)


def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=2, random_state=42)
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


def test_small_sample():
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


def test_perf():
    """Test gpu speed boost and correctness"""
    n_samples = 2000
    X = generate_cudata(n_samples, random_state=0)
    Y = generate_cudata(n_samples, random_state=0)
    Z = generate_cudata(n_samples, random_state=0)
    XYZ=pd.concat([pd.DataFrame(X),pd.DataFrame(Y),pd.DataFrame(Z)],axis=1)

    t0 = time()
    cpu_enc = GapEncoder(random_state=42, engine='cpu')
    CW=cpu_enc.fit_transform(XYZ)
    t01=time()-t0
    t1 = time()
    gpu_enc = GapEncoder(random_state=42, engine='gpu')
    GW=gpu_enc.fit_transform(XYZ)
    t02=time()-t1

    assert(t01 > t02)
    intersect=np.sum(np.sum(pd.DataFrame(CW)==pd.DataFrame(GW)))
    union=pd.DataFrame(CW).shape[0]*pd.DataFrame(CW).shape[1]
    assert(intersect==union)


def test_multiplicative_update_h_smallfast():
    # Generate random input arrays
    from scipy.sparse import csr_matrix
    from numpy.testing import assert_array_almost_equal

    np.random.seed(123)
    Vt = cp.random.rand(100, 20)
    W = cp.random.rand(20, 30)
    Ht = cp.random.rand(30, 100)
    # Convert Vt to CSR sparse matrix
    Vt_sparse = csr_matrix(Vt)
    # Call the function with different arguments
    res_1 = _multiplicative_update_h_smallfast(Vt, W, Ht, max_iter=100)
    res_2 = _multiplicative_update_h_smallfast(Vt_sparse, W, Ht, max_iter=100)
    res_3 = _multiplicative_update_h_smallfast(Vt, W, Ht, rescale_W=True, gamma_scale_prior=5.0, max_iter=200)
    # Check that the output arrays have the expected shape and type
    assert res_1.shape == (30, 100)
    assert res_2.shape == (30, 100)
    assert res_3.shape == (30, 100)
    assert res_1.dtype == cp.float32
    assert res_2.dtype == cp.float32
    assert res_3.dtype == cp.float32
    # Check that the output arrays are equal (within a small tolerance)
    assert_array_almost_equal(res_1, res_2, decimal=4)
    assert_array_almost_equal(res_1, res_3, decimal=4)


def test_multiplicative_update_w_smallfast():
    from scipy.sparse import csr_matrix
    from typing import Tuple
    from numpy.testing import assert_array_almost_equal


    # Generate random input arrays
    np.random.seed(123)
    Vt = cp.random.rand(100, 20)
    W = cp.random.rand(20, 30)
    A = cp.random.rand(20, 30)
    B = cp.random.rand(1, 30)
    Ht = cp.random.rand(30, 100)
    rescale = bool(np.random.randint(2, size=1))
    rho = float(cp.random.randn())

    # Convert Vt to CSR sparse matrix
    Vt_sparse = csr_matrix(Vt)

    # Call the function with different arguments
    res_1 = _multiplicative_update_w_smallfast(Vt, W, A, B, Ht, rescale_W=rescale, rho=rho)
    res_2 = _multiplicative_update_w_smallfast(Vt_sparse, W, A, B, Ht, rescale_W=rescale, rho=rho)

    # Check that the output arrays have the expected shape and type
    assert res_1[0].shape == W.shape
    assert res_1[1].shape == A.shape
    assert res_1[2].shape == B.shape
    assert res_1[0].dtype == cp.float32
    assert res_1[1].dtype == cp.float32
    assert res_1[2].dtype == np.float32

    assert res_2[0].shape == W.shape
    assert res_2[1].shape == A.shape
    assert res_2[2].shape == B.shape
    assert res_2[0].dtype == cp.float32
    assert res_2[1].dtype == cp.float32
    assert res_2[2].dtype == cp.float32

    # Check that the output arrays are equal (within a small tolerance)
    assert_array_almost_equal(res_1[0], res_2[0], decimal=4) # assert for W
    assert_array_almost_equal(res_1[1], res_2[1], decimal=4) # assert for A
    assert_array_almost_equal(res_1[2], res_2[1], decimal=4) # assert for B
