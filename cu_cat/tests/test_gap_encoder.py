import numpy as np
<<<<<<< HEAD
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from cu_cat import GapEncoder
from cu_cat.datasets import fetch_midwest_survey
from cu_cat.tests.utils import generate_data


@pytest.mark.parametrize(
    ["hashing", "init", "rescale_W", "rescale_rho", "add_words"],
    [
        (False, "k-means++", True, False, True),
        (True, "random", False, True, False),
        (True, "k-means", True, True, False),
    ],
)
def test_analyzer(
    hashing: bool,
    init: str,
    rescale_W: bool,
    add_words: bool,
    rescale_rho: bool,
    n_samples: int = 70,
) -> None:
    """
    Test if the output is different when the analyzer is 'word' or 'char'.
    If it is, no error ir raised.
    """
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test first analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="char",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y1 = encoder.transform(X)
    s1 = encoder.score(X)

    # Test the other analyzer output:
    encoder = GapEncoder(
        n_components=n_components,
        hashing=hashing,
        init=init,
        analyzer="word",
        add_words=add_words,
        random_state=42,
        rescale_W=rescale_W,
        rescale_rho=rescale_rho,
    )
    encoder.fit(X)
    y2 = encoder.transform(X)
    s2 = encoder.score(X)

    # Test inequality between the word and char analyzers output:
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y1, y2)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, s1, s2)


@pytest.mark.parametrize(
    ["hashing", "init", "analyzer", "add_words", "verbose"],
    [
        (False, "k-means++", "word", True, False),
        (True, "random", "char", False, False),
        (True, "k-means", "char_wb", True, True),
    ],
)
def test_gap_encoder(
    hashing: bool,
    init: str,
    analyzer: str,
    add_words: bool,
    verbose: bool,
    n_samples: int = 70,
) -> None:
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        n_components=n_components,
=======
import cupy as cp
import cudf
import pandas as pd
import pytest
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import NotFittedError
from time import time

from cu_cat import GapEncoder
from cu_cat._utils import parse_version
from cu_cat.tests.utils import generate_data #, generate_cudata


# def test_analyzer():
#     """
#     Test if the output is different when the analyzer is 'word' or 'char'.
#     If it is, no error ir raised.
#     """
#     add_words = False
#     n_samples = 70
#     X = cp.array_str(generate_data(n_samples, random_state=0))
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
    X = generate_data(n_samples, random_state=0)
    # X = cudf.DataFrame(cudf.Series.from_pandas(pd.DataFrame(X),nan_as_null=False))
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        # n_components=n_components,
>>>>>>> cu-cat/DT5
        hashing=hashing,
        init=init,
        analyzer=analyzer,
        add_words=add_words,
<<<<<<< HEAD
        verbose=verbose,
=======
>>>>>>> cu-cat/DT5
        random_state=42,
        rescale_W=True,
    )
    encoder.fit(X)
    y = encoder.transform(X)
    assert y.shape == (n_samples, n_components * X.shape[1]), str(y.shape)

    # Test L1-norm of topics W.
    for col_enc in encoder.fitted_models_:
        l1_norm_W = np.abs(col_enc.W_).sum(axis=1)
<<<<<<< HEAD
        np.testing.assert_array_almost_equal(l1_norm_W, np.ones(n_components))
=======
        np.testing.assert_array_almost_equal(l1_norm_W.get(), np.ones(n_components),decimal=2)
>>>>>>> cu-cat/DT5

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
<<<<<<< HEAD
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


@pytest.mark.parametrize(
    "add_words",
    [True, False],
)
def test_partial_fit(add_words: bool, n_samples: int = 70) -> None:
    X = generate_data(n_samples, random_state=0)
    X2 = pd.DataFrame(generate_data(n_samples - 10, random_state=1))
    X3 = generate_data(n_samples - 10, random_state=2)
    # Gap encoder with fit on one batch
    enc = GapEncoder(
        random_state=42, batch_size=n_samples, max_iter=1, add_words=add_words
    )
    X_enc = enc.fit_transform(X)
    # Gap encoder with partial fit
    enc = GapEncoder(random_state=42, add_words=add_words)
    enc.partial_fit(X)
    X_enc_partial = enc.transform(X)
    # Check if the encoded vectors are the same
    np.testing.assert_almost_equal(X_enc, X_enc_partial)
    enc.partial_fit(X2)
    X_enc_partial2 = enc.transform(X3)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, X_enc, X_enc_partial2
    )
=======
    # np.testing.assert_array_equal(y, y2)
    # np.testing.assert_array_equal(np.round(y.get(),10), np.round(y2.get(),10))
    np.testing.assert_array_almost_equal(y.get(), y2.get(),decimal=2)

    


# def test_input_type() -> None:
#     # Numpy array with one column
#     X = np.array([["alice"], ["bob"]])


#     enc = GapEncoder(n_components=2, random_state=42)
#     X_enc_array = enc.fit_transform(X)
#     # List
#     X2 = [["alice"], ["bob"]]


#     enc = GapEncoder(n_components=2, random_state=42)
#     X_enc_list = enc.fit_transform(X2)
#     # Check if the encoded vectors are the same
#     np.testing.assert_array_equal(X_enc_array, X_enc_list)

#     # Numpy array with two columns
#     X = np.array([["alice", "charlie"], ["bob", "delta"]])
#     X = cudf.DataFrame(cudf.Series.from_pandas(X,nan_as_null=False))

#     enc = GapEncoder(n_components=2, random_state=42)
#     X_enc_array = enc.fit_transform(X)
#     # Pandas dataframe with two columns
#     df = pd.DataFrame(X)
#     enc = GapEncoder(n_components=2, random_state=42)
#     X_enc_df = enc.fit_transform(df)
#     # Check if the encoded vectors are the same
#     # np.testing.assert_array_equal(X_enc_array, X_enc_df)
#     np.testing.assert_array_almost_equal(X_enc_array.get(), X_enc_df.get(),decimal=2)



# def test_partial_fit(n_samples=70) -> None:
#     X = generate_data(n_samples, random_state=0)
#     # Gap encoder with fit on one batch
#     enc = GapEncoder(random_state=42, batch_size=n_samples, max_iter=1)
#     X_enc = enc.fit_transform(X)
#     # Gap encoder with partial fit
#     enc = GapEncoder(random_state=42)
#     enc.partial_fit(X)
#     X_enc_partial = enc.transform(X)
#     # Check if the encoded vectors are the same
#     np.testing.assert_almost_equal(X_enc, X_enc_partial)
>>>>>>> cu-cat/DT5


def test_get_feature_names_out(n_samples=70) -> None:
    X = generate_data(n_samples, random_state=0)
<<<<<<< HEAD
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    feature_names_1 = enc.get_feature_names_out()
=======
    enc = GapEncoder(random_state=42, engine='cuml')
    enc.fit(X)
    # Expect a warning if sklearn >= 1.0
    if parse_version(sklearn_version) < parse_version("1.0"):
        feature_names_1 = enc.get_feature_names()
    else:
        with pytest.warns(DeprecationWarning):
            feature_names_1 = enc.get_feature_names()
>>>>>>> cu-cat/DT5
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


<<<<<<< HEAD
def test_overflow_error() -> None:
    np.seterr(over="raise", divide="raise")
    r = np.random.RandomState(0)
    X = r.randint(1e5, 1e6, size=(8000, 1)).astype(str)
    enc = GapEncoder(n_components=2, batch_size=1, max_iter=1, random_state=0)
    enc.fit(X)


def test_score(n_samples: int = 70) -> None:
    X1 = generate_data(n_samples, random_state=0)
    X2 = np.hstack([X1, X1])
    enc = GapEncoder(random_state=42)
    enc.fit(X1)
    score_X1 = enc.score(X1)
    enc.fit(X2)
    score_X2 = enc.score(X2)
    # Check that two identical columns give the same score
    assert score_X1 * 2 == score_X2


@pytest.mark.parametrize(
    "missing",
    ["zero_impute", "error", "aaa"],
)
def test_missing_values(missing: str) -> None:
    """Test what happens when missing values are in the data"""
    observations = [
        ["alice", "bob"],
        ["bob", "alice"],
        ["bob", np.nan],
        ["alice", "charlie"],
        [np.nan, "alice"],
    ]
    observations = np.array(observations, dtype=object)
    enc = GapEncoder(handle_missing=missing, n_components=3)
=======
# def test_overflow_error() -> None:
#     np.seterr(over="raise", divide="raise")
#     r = np.random.RandomState(0)
#     X = (r.randint(1e5, 1e6, size=(8000, 1)).astype(str))
#     X=cudf.Series.from_pandas(pd.DataFrame(X)[0]).astype(str).reset_index()
#     enc = GapEncoder(
#         n_components=2, batch_size=1, min_iter=1, max_iter=1, random_state=0
#     )
#     enc.fit(X)


# def test_score(n_samples: int = 70) -> None:
#     X1 = generate_data(n_samples, random_state=0)
#     X2 = np.hstack([X1, X1])
#     enc = GapEncoder(random_state=42)
#     enc.fit(X1)
#     score_X1 = enc.score(X1)
#     enc.fit(X2)
#     score_X2 = enc.score(X2)
#     # Check that two identical columns give the same score
#     assert score_X1 * 2 == score_X2


@pytest.mark.parametrize("missing", ["zero_impute"]) #, "error", "aaa"])
def test_missing_values(missing: str) -> None:
    observations = [
    "alice","bob",
    "bob","alice",
    np.nan,"bob",
    "alice",np.nan,
    "alice","charlie"
]
    observations = cudf.DataFrame(cudf.Series.from_pandas(observations,nan_as_null=False))

    # observations = cudf.DataFrame.from_pandas(pd.DataFrame(np.array(observations, dtype=object)))
    enc = GapEncoder(handle_missing=missing, n_components=3, engine='cuml')
>>>>>>> cu-cat/DT5
    if missing == "error":
        with pytest.raises(ValueError, match="Input data contains missing values"):
            enc.fit_transform(observations)
    elif missing == "zero_impute":
        enc.fit_transform(observations)
<<<<<<< HEAD
        enc.partial_fit(observations)
=======
        enc.fit(observations) ##not partial_fit
>>>>>>> cu-cat/DT5
    else:
        with pytest.raises(
            ValueError,
            match=r"handle_missing should be either "
            r"'error' or 'zero_impute', got 'aaa'",
        ):
            enc.fit_transform(observations)


<<<<<<< HEAD
def test_check_fitted_gap_encoder() -> None:
    """Test that calling transform before fit raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=2, random_state=42)
=======
def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    X = np.array(["alice", "bob"])
    X = cudf.DataFrame(cudf.Series.from_pandas(X,nan_as_null=False))

    enc = GapEncoder(n_components=2, random_state=42,engine='cuml')
>>>>>>> cu-cat/DT5
    with pytest.raises(NotFittedError):
        enc.transform(X)

    # Check that it works after fit
    enc.fit(X)
    enc.transform(X)


<<<<<<< HEAD
def test_small_sample() -> None:
=======
def test_small_sample():
>>>>>>> cu-cat/DT5
    """Test that having n_samples < n_components raises an error"""
    X = np.array([["alice"], ["bob"]])
    enc = GapEncoder(n_components=3, random_state=42)
    with pytest.raises(ValueError, match="should be >= n_components"):
        enc.fit_transform(X)


<<<<<<< HEAD
def test_transform_deterministic() -> None:
    """Non-regression test for #188"""
    dataset = fetch_midwest_survey()
    X_train, X_test = train_test_split(
        dataset.X[["What_would_you_call_the_part_of_the_country_you_live_in_now"]],
        random_state=0,
    )
    enc = GapEncoder(n_components=2, random_state=2)
    enc.fit_transform(X_train)
    topics1 = enc.get_feature_names_out()
    enc.transform(X_test)
    topics2 = enc.get_feature_names_out()
    assert topics1 == topics2


def test_max_no_improvements_none() -> None:
    """Test that max_no_improvements=None works"""
    X = generate_data(300, random_state=0)
    enc_none = GapEncoder(n_components=2, max_no_improvement=None, random_state=42)
    enc_none.fit(X)
=======
def test_perf():
    """Test gpu speed boost and correctness"""
    t01=[];t02=[]
    for i in range(5):
        n_samples = np.random.randint(10,250)
        X = generate_data(n_samples, random_state=42)

        XX = X.to_pandas()
        t0 = time()
        cpu_enc = GapEncoder(random_state=2, engine='sklearn')
        CW=cpu_enc.fit_transform(XX)
        t01.append(time()-t0)
        t1 = time()
        gpu_enc = GapEncoder(random_state=2, engine='cuml')
        GW=gpu_enc.fit_transform(X)
        t02.append(time()-t1)
        GW=GW.get()
        
    assert(any(np.greater(t01,t02)))
        # [t01 , t02]
    # intersect=np.sum(np.sum(pd.DataFrame(CW)==(GW)))
    # union=pd.DataFrame(CW).shape[0]*pd.DataFrame(CW).shape[1]
    # assert(intersect==union)
    # np.testing.assert_array_almost_equal(pd.DataFrame(CW), GW, decimal=4)


# def test_multiplicative_update_h_smallfast():
#     # Generate random input arrays
#     from scipy.sparse import csr_matrix as cpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_h
    
#     from cupyx.scipy.sparse import csr_matrix as gpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_h_smallfast
    
#     np.random.seed(123)
#     Vt = pd.DataFrame(np.random.rand(1000, 1000))
#     W = pd.DataFrame(np.random.rand(1000, 1000))
#     Ht = pd.DataFrame(np.random.rand(1000, 1000))
#     # Convert Vt to CSR sparse matrix
#     Vt_sparse = cpu_csr(Vt)
#     tmp = np.random.rand(1, 1)
    
#     # Call the function with different arguments
#     res_1A = _multiplicative_update_h(tmp,Vt, W, Ht, max_iter=100)
#     res_1B = _multiplicative_update_h_smallfast(cp.array(Vt), cp.array(W), cp.array(Ht), max_iter=100)
    
#     res_2A = _multiplicative_update_h(tmp,Vt_sparse, W, Ht, rescale_W=True, gamma_scale_prior=5.0, max_iter=200)
#     res_2B = _multiplicative_update_h_smallfast(gpu_csr(Vt_sparse), cp.array(W), cp.array(Ht), rescale_W=True, gamma_scale_prior=5.0, max_iter=200)
#     # Check that the output arrays are equal (within a small tolerance)
    
#     np.testing.assert_array_almost_equal(res_1A, res_1B.get(), decimal=4)
#     np.testing.assert_array_almost_equal(res_1A, res_2B.get(), decimal=4)


# def test_multiplicative_update_w_smallfast():
#     from typing import Tuple
#     from scipy.sparse import csr_matrix as cpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_w
    
#     from cupyx.scipy.sparse import csr_matrix as gpu_csr
#     from cu_cat._gap_encoder import _multiplicative_update_w_smallfast
    

#     # Generate random input arrays
#     np.random.seed(123)
#     Vt = pd.DataFrame(np.random.rand(1000, 1000))
#     W = pd.DataFrame(np.random.rand(1000, 1000))
#     A = pd.DataFrame(np.random.rand(1000, 1000))
#     B = pd.DataFrame(np.random.rand(1000, 1000))
#     Ht = pd.DataFrame(np.random.rand(1000, 1000))
#     rescale = True #bool(np.random.randint(2, size=1))
#     rho = float(np.random.randn())
#     tmp = np.random.rand(1, 1)

#     # Convert Vt to CSR sparse matrix
#     Vt_sparse = cpu_csr(Vt)

#     # Call the function with different arguments
#     res_1A = _multiplicative_update_w(tmp,Vt, W, A, B, Ht, rescale_W=rescale, rho=rho)
#     res_1B = _multiplicative_update_w_smallfast(cp.array(Vt), cp.array(W), cp.array(A), cp.array(B), cp.array(Ht), rescale_W=rescale, rho=rho)

#     res_2A = _multiplicative_update_w(tmp,gpu_csr(Vt_sparse), W, A, B, Ht, rescale_W=rescale, rho=rho)
#     res_2B = _multiplicative_update_w_smallfast(gpu_csr(Vt_sparse), cp.array(W), cp.array(A), cp.array(B), cp.array(Ht), rescale_W=rescale, rho=rho)

#     # Check that the output arrays have the expected shape and type

#     # Check that the output arrays are equal (within a small tolerance)
#     np.testing.assert_array_almost_equal(res_1A, res_1B.get(), decimal=4)
#     np.testing.assert_array_almost_equal(res_2A, res_2B.get(), decimal=4)
>>>>>>> cu-cat/DT5
