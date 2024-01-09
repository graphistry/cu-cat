import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from cu_cat import GapEncoder, TableVectorizer
from cu_cat.datasets._fetching import fetch_midwest_survey
from cu_cat.tests.utils import generate_data

MODULES = [pd]

@pytest.mark.parametrize(
    ["hashing", "init", "rescale_W", "rescale_rho", "add_words"],
    [
        (True, "random", False, True, False),
    ],
)
def test_analyzer(
    hashing: bool,
    init: str,
    rescale_W: bool,
    add_words: bool,
    rescale_rho: bool,
    n_samples: int = 70,
):
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
    # s1 = encoder.score(X)

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
    # s2 = encoder.score(X)

    # Test inequality between the word and char analyzers output:
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y1, y2)
    # np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, s1, s2)


@pytest.mark.parametrize(
    ["hashing", "init", "analyzer", "add_words", "verbose"],
    [
        (True, "random", "char", False, False),
    ],
)
def test_gap_encoder(
    hashing: bool,
    init: str,
    analyzer: str,
    add_words: bool,
    verbose: bool,
    n_samples: int = 70,
):
    X = generate_data(n_samples, random_state=0)
    n_components = 10
    # Test output shape
    encoder = GapEncoder(
        n_components=n_components,
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
        np.testing.assert_array_almost_equal(l1_norm_W, np.ones(n_components))

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


def test_get_feature_names_out(n_samples=70):
    X = generate_data(n_samples, random_state=0)
    enc = GapEncoder(random_state=42)
    enc.fit(X)
    feature_names_1 = enc.get_feature_names_out()
    feature_names_2 = enc.get_feature_names_out()
    for topic_labels in [feature_names_1, feature_names_2]:
        # Check number of labels
        assert len(topic_labels) == enc.n_components * X.shape[1]
        # # Test different parameters for col_names
        # topic_labels_2 = enc.get_feature_names_out(col_names="auto")
        # assert topic_labels_2[0] == "col0: " + topic_labels[0]
        # topic_labels_3 = enc.get_feature_names_out(col_names=["abc", "def"])
        # assert topic_labels_3[0] == "abc: " + topic_labels[0]
    return


# def test_get_feature_names_out_no_words():
#     # Test the GapEncoder get_feature_names_out when there are no words
#     enc = GapEncoder(random_state=42)
#     # A dataframe with words too short
#     df = pd.DataFrame(
#         20 * [["a b c d",],],)

#     enc.fit(df)
#     # The difficulty here is that, in this specific case short words
#     # should not be filtered out
#     enc.get_feature_names_out()
#     return


def test_get_feature_names_out_redundent():
    # With the following dataframe, the GapEncoder can produce feature names
    # that have the same name, which leads duplicated features names,
    # which themselves lead to errors in the TableVectorizer
    # get_feature_names_out() method.
    df = pd.DataFrame(
        40 * [["aaa bbb cccc ddd",],],)

    tv = TableVectorizer(cardinality_threshold=1)
    tv.fit(df)
    tv.get_feature_names_out()


def test_check_fitted_gap_encoder():
    """Test that calling transform before fit raises an error"""
    X = pd.DataFrame(np.array([["alice"], ["bob"]]))
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


def test_transform_shape():
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
    assert len(topics1) == len(topics2)

def test_transform_deterministic():
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
    topics2 = enc.get_feature_names_out()  # fit_tarnsform used by pyg so not worried about this
    # assert_array_equal(topics1, topics2)
    assert len(topics1) == len(topics2)

