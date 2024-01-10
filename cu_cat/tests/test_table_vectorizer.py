from __future__ import annotations
import joblib, subprocess
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils._testing import assert_array_equal, skip_if_no_parallel
from sklearn.utils.validation import check_is_fitted

from cu_cat._gap_encoder import GapEncoder
from cu_cat._table_vectorizer import _infer_date_format, TableVectorizer
from cu_cat.tests.utils import transformers_list_equal
from cu_cat._dep_manager import deps
dirty_cat = deps.dirty_cat


MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"


def check_same_transformers(
    expected_transformers: dict, actual_transformers: list
) -> None:
    # Construct the dict from the actual transformers
    actual_transformers_dict = {name: cols for name, trans, cols in actual_transformers}
    assert actual_transformers_dict == expected_transformers


def type_equality(expected_type, actual_type) -> bool:
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the TableVectorizer).
    """
    if (isinstance(expected_type, object) or isinstance(expected_type, str)) and (
        isinstance(actual_type, object) or isinstance(actual_type, str)
    ):
        return True
    else:
        return expected_type == actual_type


def _get_clean_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, 63, 12, 44], dtype="int"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, 9.0], dtype="float"),
            "str1": pd.Series(
                ["public", "private", "private", "private", "public"], dtype="string"
            ),
            "str2": pd.Series(
                ["officer", "manager", "lawyer", "chef", "teacher"], dtype="string"
            ),
            "cat1": pd.Series(["yes", "yes", "no", "yes", "no"], dtype="category"),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", "50K+"], dtype="category"
            ),
        }
    )


def _get_dirty_dataframe(categorical_dtype="object") -> pd.DataFrame:
    """
    Creates a simple DataFrame with some missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to test the robustness of the vectorizer.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, pd.NA, 12, 44], dtype="Int64"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, np.nan], dtype="Float64"),
            "str1": pd.Series(
                ["public", np.nan, "private", "private", "public"],
                dtype=categorical_dtype,
            ),
            "str2": pd.Series(
                ["officer", "manager", None, "chef", "teacher"],
                dtype=categorical_dtype,
            ),
            "cat1": pd.Series(
                [np.nan, "yes", "no", "yes", "no"], dtype=categorical_dtype
            ),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", np.nan], dtype=categorical_dtype
            ),
        }
    )


def _get_mixed_types_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int_str": ["1", "2", 3, "3", 5],
            "float_str": ["1.0", pd.NA, 3.0, "3.0", 5.0],
            "int_float": [1, 2, 3.0, 3, 5.0],
            "bool_str": ["True", False, True, "False", "True"],
        }
    )


def _get_mixed_types_array() -> np.ndarray:
    return np.array(
        [
            ["1", "2", 3, "3", 5],
            ["1.0", np.nan, 3.0, "3.0", 5.0],
            [1, 2, 3.0, 3, 5.0],
            ["True", False, True, "False", "True"],
        ]
    ).T

def _get_list_of_lists() -> list:
    return _get_numpy_array().tolist()



def _test_possibilities(X) -> None:
    """
    Do a bunch of tests with the TableVectorizer.
    We take some expected transformers results as argument. They're usually
    lists or dictionaries.
    """
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    expected_transformers_df = {
        "numeric": ["int", "float"],
        "low_cardinality": ["str1", "cat1"],
        "high_cardinality": ["str2", "cat2"],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers_)

    # Test with higher cardinality threshold and no numeric transformer
    expected_transformers_2 = {
        "low_cardinality": ["str1", "str2", "cat1", "cat2"],
        "numeric": ["int", "float"],
    }
    vectorizer_default = TableVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers_)

    # Test with single column dataframe
    expected_transformers_series = {
        "low_cardinality": ["cat1"],
    }
    vectorizer_base.fit_transform(X[["cat1"]])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers_)

    # Test casting values
    vectorizer_cast = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    X_str = X.astype("object")
    # With pandas
    expected_transformers_plain = {
        "high_cardinality": ["str2", "cat2"],
        "low_cardinality": ["str1", "cat1"],
        "numeric": ["int", "float"],
    }
    vectorizer_cast.fit_transform(X_str)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers_)
    # With numpy
    expected_transformers_np_cast = {
        "numeric": [0, 1],
        "low_cardinality": [2, 4],
        "high_cardinality": [3, 5],
    }
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(
        expected_transformers_np_cast, vectorizer_cast.transformers_
    )

def test_with_clean_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a clean dataset.
    """
    _test_possibilities(_get_clean_dataframe())


def test_with_dirty_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a dataset containing missing values.
    """
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="object"))
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="category"))


def test_get_feature_names_out() -> None:
    X = _get_clean_dataframe()

    vec_w_pass = TableVectorizer(remainder="passthrough")
    vec_w_pass.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_pass = np.array(
        [
            "int",
            "float",
            "str1_public",
            "str2_chef",
            "str2_lawyer",
            "str2_manager",
            "str2_officer",
            "str2_teacher",
            "cat1_yes",
            "cat2_20K+",
            "cat2_30K+",
            "cat2_40K+",
            "cat2_50K+",
            "cat2_60K+",
        ]
    )
    assert len(vec_w_pass.get_feature_names_out()) >= len(expected_feature_names_pass)

    vec_w_drop = TableVectorizer(remainder="drop")
    vec_w_drop.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_drop = [
        "int",
        "float",
        "str1_public",
        "str2_chef",
        "str2_lawyer",
        "str2_manager",
        "str2_officer",
        "str2_teacher",
        "cat1_yes",
        "cat2_20K+",
        "cat2_30K+",
        "cat2_40K+",
        "cat2_50K+",
        "cat2_60K+",
    ]
    assert len(vec_w_drop.get_feature_names_out()) >= len(expected_feature_names_drop)


def test_fit() -> None:
    # Simply checks sklearn's `check_is_fitted` function raises an error if
    # the TableVectorizer is instantiated but not fitted.
    # See GH#193
    table_vec = TableVectorizer()
    with pytest.raises(NotFittedError):
        assert check_is_fitted(table_vec)



# def test_fit_transform_equiv() -> None:
#     """
#     We will test the equivalence between using `.fit_transform(X)`
#     and `.fit(X).transform(X).`
#     """
#     for X in [
#         _get_clean_dataframe(),
#         _get_dirty_dataframe(categorical_dtype="object"),
#         _get_dirty_dataframe(categorical_dtype="category"),
#         _get_mixed_types_dataframe(),
#         _get_mixed_types_array(),
#     ]:
#         enc1_x1 = TableVectorizer().fit_transform(X)
#         enc2_x1 = TableVectorizer().fit(X).transform(X)

#         assert np.allclose(enc1_x1, enc2_x1, rtol=0, atol=0, equal_nan=True)


def test_check_fitted_table_vectorizer() -> None:
    """Test that calling transform before fit raises an error"""
    X = _get_clean_dataframe()
    tv = TableVectorizer()
    with pytest.raises(NotFittedError):
        tv.transform(X)

    # Test that calling transform after fit works
    tv.fit(X)
    tv.transform(X)


# def test_deterministic(pipeline) -> None:
#     """
#     Tests that running the same TableVectorizer multiple times with the same
#     (deterministic) components results in the same output.
#     """
#     X = _get_dirty_dataframe()
#     X_enc_prev = pipeline.fit_transform(X)
#     for _ in range(5):
#         X_enc = pipeline.fit_transform(X)
#         np.testing.assert_array_equal(X_enc, X_enc_prev)
#         X_enc_prev = X_enc


# def test_mixed_types() -> None:
#     # TODO: datetime/str mixed types
#     # don't work
#     df = _get_mixed_types_dataframe()
#     table_vec = TableVectorizer()
#     table_vec.fit_transform(df)
#     # check that the types are correctly inferred
#     table_vec.fit_transform(df)
#     expected_transformers_df = {
#         "numeric": ["int_str", "float_str", "int_float"],
#         "low_card_cat": ["bool_str"],
#     }
#     check_same_transformers(expected_transformers_df, table_vec.transformers_)



# def test_changing_types_int_float() -> None:
#     # The TableVectorizer shouldn't cast floats to ints
#     # even if only ints were seen during fit
#     X_fit, X_transform = (
#         pd.DataFrame(pd.Series([1, 2, 3])),
#         pd.DataFrame(pd.Series([1, 2, 3.3])),
#     )
#     table_vec = TableVectorizer()
#     table_vec.fit_transform(X_fit)
#     res = table_vec.transform(X_transform)
#     assert np.allclose(res, np.array([[1.0], [2.0], [3.3]]))



def test_HN():
    # from cu_cat import TableVectorizer
    # import pandas as pd
    askHN = pd.read_csv('https://storage.googleapis.com/cohere-assets/blog/text-clustering/data/askhn3k_df.csv', index_col=0)
    table_vec = TableVectorizer()
    aa = table_vec.fit_transform((askHN))
    if deps.dirty_cat:
        bb = dirty_cat.TableVectorizer().fit_transform(askHN)
        assert aa.shape[0] == bb.shape[0]
    else:
        assert aa.shape[0] == askHN.shape[0]

def test_red_team():
    df = pd.read_csv('https://gist.githubusercontent.com/silkspace/c7b50d0c03dc59f63c48d68d696958ff/raw/31d918267f86f8252d42d2e9597ba6fc03fcdac2/redteam_50k.csv', index_col=0)
    red_team = pd.read_csv('https://gist.githubusercontent.com/silkspace/5cf5a94b9ac4b4ffe38904f20d93edb1/raw/888dabd86f88ea747cf9ff5f6c44725e21536465/redteam_labels.csv', index_col=0)
    df['feats'] = df.src_computer + ' ' + df.dst_computer + ' ' + df.auth_type + ' ' + df.logontype
    df['feats2'] = df.src_computer + ' ' + df.dst_computer
    ndf = df.drop_duplicates(subset=['feats'])
    tdf = pd.concat([red_team.reset_index(), ndf.reset_index()])
    tdf['node'] = range(len(tdf))
    table_vec = TableVectorizer()
    aa = table_vec.fit_transform((tdf))
    if deps.dirty_cat:
        bb = dirty_cat.TableVectorizer().fit_transform(tdf)
        assert aa.shape[0] == bb.shape[0]
    else:
        assert aa.shape[0] == tdf.shape[0]


def test_malware():
    edf = pd.read_csv('https://gist.githubusercontent.com/silkspace/33bde3e69ae24fee1298a66d1e00b467/raw/dc66bd6f1687270be7098f94b3929d6a055b4438/malware_bots.csv', index_col=0)
    T = edf.Label.apply(lambda x: True if 'Botnet' in x else False)
    bot = edf[T]
    nbot = edf[~T]
    print(f'Botnet abundance: {100*len(bot)/len(edf):0.2f}%')# so botnet traffic makes up a tiny fraction of total

    # let's balance the dataset in a 10-1 ratio, for speed and demonstrative purposes
    negs = nbot.sample(10*len(bot))
    edf = pd.concat([bot, negs])  # top part of arrays are bot traffic, then all non-bot traffic
    edf = edf.drop_duplicates()
    table_vec = TableVectorizer()
    aa = table_vec.fit_transform((edf))
    if deps.dirty_cat:
        bb = dirty_cat.TableVectorizer().fit_transform(edf)
        assert aa.shape[0] == bb.shape[0]
    else:
        assert aa.shape[0] == edf.shape[0]

def test_20newsgroups():
    from sklearn.datasets import fetch_20newsgroups
    n_samples = 1000

    news, _ = fetch_20newsgroups(
        shuffle=True,
        random_state=1,
        remove=("headers", "footers", "quotes"),
        return_X_y=True,
    )

    news = news[:n_samples]
    news=pd.DataFrame(news)
    table_vec = TableVectorizer()
    aa = table_vec.fit_transform((news))
    if deps.dirty_cat:
        bb = dirty_cat.TableVectorizer().fit_transform(news)
        assert aa.shape[0] == bb.shape[0]
    else:
        assert aa.shape[0] == news.shape[0]
