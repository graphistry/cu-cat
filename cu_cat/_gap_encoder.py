"""
<<<<<<<< HEAD:skrub/_gap_encoder.py
Implements the GapEncoder: a probabilistic encoder for categorical variables.
"""

from collections.abc import Generator
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
========
Online Gamma-Poisson factorization of string arrays.
The principle is as follows:
    1. Given an input string array X, we build its bag-of-n-grams
       representation V (n_samples, vocab_size).
    2. Instead of using the n-grams counts as encodings, we look for low-
       dimensional representations by modeling n-grams counts as linear
       combinations of topics V = HW, with W (n_topics, vocab_size) the topics
       and H (n_samples, n_topics) the associated activations.
    3. Assuming that n-grams counts follow a Poisson law, we fit H and W to
       maximize the likelihood of the data, with a Gamma prior for the
       activations H to induce sparsity.
    4. In practice, this is equivalent to a non-negative matrix factorization
       with the Kullback-Leibler divergence as loss, and a Gamma prior on H.
       We thus optimize H and W with the multiplicative update method.

Depending on available GPU memory, GapEncoder will attempt to fit NMF via matrix multiplication at once rather than serializing dot products.
This dynamicity is accomplished through get_gpu_memory() and the gmem variable, initiated during lazy cuml import.

If the gpu memory is too small, it will default to serializing dot products on the GPU, and in some (rare) cases the CPU will be faster until sufficient samples are provided to outcompete the parallelization of the CPU loops across features.
"""

import warnings,sys
from typing import Dict, Generator, List, Literal, Optional, Tuple, Union
from inspect import getmodule
import cupy as cp, cudf, pyarrow, cuml
import numpy as np
import pandas as pd
from time import time

from cupyx.scipy.sparse import csr_matrix as csr_gpu
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
<<<<<<<< HEAD:skrub/_gap_encoder.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
========
from scipy.sparse import csr_matrix as csr_cpu
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator, TransformerMixin
from cuml.cluster import KMeans
# from cuml.feature_extraction.text import CountVectorizer,HashingVectorizer
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.fixes import _object_dtype_isnan
from sklearn.utils.validation import _num_samples, check_is_fitted

<<<<<<<< HEAD:skrub/_gap_encoder.py
from ._utils import check_input
========
from ._utils import check_input, parse_version, df_type
import logging

if parse_version(sklearn_version) < parse_version("0.24"):
    from sklearn.cluster._kmeans import _k_init
else:
    from sklearn.cluster import kmeans_plusplus

from sklearn.decomposition._nmf import _beta_divergence

# Ignore lines too long, as some things in the docstring cannot be cut.
# flake8: noqa: E501'

logger = logging.getLogger()

def lazy_cuml_import_has_dependancy():
    try:
        import warnings

        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import cuml  # type: ignore
            import subprocess as sp
            import os

            def get_gpu_memory():
                command = "nvidia-smi --query-gpu=memory.free --format=csv"
                memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
                memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
                return memory_free_values

            get_gpu_memory()
            # from cuml.feature_extraction.text import CountVectorizer,HashingVectorizer


        return True, "ok", cuml,get_gpu_memory() #,CountVectorizer,HashingVectorizer
    except ModuleNotFoundError as e:
        return False, e, None, None
    
def lazy_sklearn_import_has_dependancy():
    try:
        import warnings

        warnings.filterwarnings("ignore")
        import sklearn  # noqa
        # from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer


        return True, "ok", sklearn #,CountVectorizer,HashingVectorizer
    except ModuleNotFoundError as e:
        return False, e, None


def assert_imported():
    has_dependancy_, import_exn, _ = lazy_sklearn_import_has_dependancy()
    if not has_dependancy_:
        logger.error("SKLEARN not found, trying running " "`pip install sklearn`")
        raise import_exn


def assert_imported_cuml():
    has_cuml_dependancy_, import_cuml_exn, _, _ = lazy_cuml_import_has_dependancy()
    if not has_cuml_dependancy_:
        logger.warning("cuML not found, trying running " "`pip install cuml`")
        raise import_cuml_exn


EngineConcrete = Literal['cuml', 'sklearn']
Engine = Literal[EngineConcrete, "auto"]


def resolve_engine(
    engine: Engine,
) -> EngineConcrete:  # noqa
    if engine in ['cuml', 'sklearn']:
        return engine  # type: ignore
    if engine in ["auto"]:
        has_cuml_dependancy_, _, _, _ = lazy_cuml_import_has_dependancy()
        if has_cuml_dependancy_:
            return 'cuml'
        has_sklearn_dependancy_, _, _ = lazy_sklearn_import_has_dependancy()
        if has_sklearn_dependancy_:
            return 'sklearn'

    raise ValueError(  # noqa
        f'engine expected to be "auto", '
        '"sklearn", or  "cuml" '
        f"but received: {engine} :: {type(engine)}"
    )
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py


class GapEncoderColumn(BaseEstimator, TransformerMixin):

    """GapEncoder for encoding a single column.

    Do not use directly, this is an internal object.

    See Also
    --------
    GapEncoder
        For more information.
    """

    rho_: float
<<<<<<<< HEAD:skrub/_gap_encoder.py
    H_dict_: dict[NDArray, NDArray]
========
    H_dict_: Dict[pyarrow.StringScalar, cp.ndarray]
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py

    def __init__(
        self,
        n_components: int = 10,
        batch_size: int = 1024,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
        rho: float = 0.95,
        rescale_rho: bool = False,
        hashing: bool = False,
        hashing_n_features: int = 2**12,
<<<<<<<< HEAD:skrub/_gap_encoder.py
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
========
        init: Literal["k-means++", "random", "k-means"] = "random",
        tol: float = 1e-4,
        min_iter: int = 2,
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        max_iter: int = 5,
        ngram_range: tuple[int, int] = (2, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char",
        add_words: bool = False,
        random_state: int | RandomState | None = None,
        rescale_W: bool = True,
<<<<<<<< HEAD:skrub/_gap_encoder.py
        max_iter_e_step: int = 1,
        max_no_improvement: int = 5,
        verbose: int = 0,
========
        max_iter_e_step: int = 20,
        engine: Engine = "auto",
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    ):
        
        engine_resolved = resolve_engine(engine)
        # FIXME remove as set_new_kwargs will always replace?
        if engine_resolved == 'sklearn':
            _, _, engine = lazy_sklearn_import_has_dependancy()
            from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
        elif engine_resolved == 'cuml':
            _, _, engine, gmem = lazy_cuml_import_has_dependancy()
            from cuml.feature_extraction.text import CountVectorizer,HashingVectorizer
        
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = check_random_state(random_state)
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step
<<<<<<<< HEAD:skrub/_gap_encoder.py
        self.max_no_improvement = max_no_improvement
        self.verbose = verbose
========
        self._CV = CountVectorizer
        self._HV = HashingVectorizer
        self.engine = engine_resolved
        self.gmem = gmem[0]
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py

    def _init_vars(self, X) -> tuple[NDArray, NDArray, NDArray]:
        """
        Build the bag-of-n-grams representation `V` of `X` and initialize
        the topics `W`.
        """
        self.Xt_ = df_type(X)
        cuml.set_global_output_type('cupy')
        # Init n-grams counts vectorizer
        if self.hashing:
            self.ngrams_count_ = self._HV(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.hashing_n_features,
                norm=None,
                alternate_sign=False,
            )
            if self.add_words:  # Init a word counts vectorizer if needed
                self.word_count_ = self._HV(
                    analyzer="word",
                    n_features=self.hashing_n_features,
                    norm=None,
                    alternate_sign=False,
                )
        else:
            self.ngrams_count_ = self._CV(
                analyzer=self.analyzer, ngram_range=self.ngram_range, dtype=np.float64
            )
            if self.add_words:
                self.word_count_ = self._CV(dtype=np.float64)

        # Init H_dict_ with empty dict to train from scratch
        self.H_dict_ = dict() #cudf.Series()
        # self.X_dict_ = cudf.Series()

        # Build the n-grams counts matrix unq_V on unique elements of X
        if 'cudf' not in str(getmodule(X)):
            unq_X, lookup = np.unique(X, return_inverse=True)
        elif 'cudf' in str(getmodule(X)):
            unq_X = X.unique()
            tmp, lookup = np.unique(X.to_arrow(), return_inverse=True)
            # unq_X = cudf.Series(unq_X)
        unq_V = self.ngrams_count_.fit_transform(unq_X)
        if self.add_words:  # Add word counts to unq_V
            unq_V2 = self.word_count_.fit_transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        if not self.hashing:  # Build n-grams/word vocabulary
<<<<<<<< HEAD:skrub/_gap_encoder.py
            self.vocabulary = self.ngrams_count_.get_feature_names_out()
            if self.add_words:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count_.get_feature_names_out())
========
            # if parse_version(sklearn_version) < parse_version("1.0"):
            #     self.vocabulary = self.ngrams_count_.get_feature_names()
            # else:
            self.vocabulary = self.ngrams_count_.get_feature_names()
            if self.add_words:
                # if parse_version(sklearn_version) < parse_version("1.0"):
                #     self.vocabulary = np.concatenate(
                #         (self.vocabulary, self.word_count_.get_feature_names())
                #     )
                # else:
                self.vocabulary = np.concatenate(
                    (self.vocabulary, self.word_count_.get_feature_names())
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
                )
        _, self.n_vocab = unq_V.shape
        # Init the topics W given the n-grams counts V

        self.W_, self.A_, self.B_ = self._init_w(unq_V[lookup], X)
        # Init the activations unq_H of each unique input string
        unq_H = _rescale_h(self, unq_V, np.ones((len(unq_X), self.n_components)))
        # Update self.H_dict_ with unique input strings and their activations
        if self.engine == 'cuml' :
            self.H_dict_.update(zip(unq_X.to_arrow(), unq_H.values))
        else:
            self.H_dict_.update(zip(unq_X, unq_H))
        if self.rescale_rho:
            # Make update rate per iteration independent of the batch_size
            self.rho_ = self.rho ** (self.batch_size / len(X))
        return unq_X, unq_V, lookup

    def _get_H(self, X: NDArray) -> NDArray:
        """
        Return the bag-of-n-grams representation of `X`.
        """
        if 'cudf' in df_type(X) or 'arrow' in df_type(self.H_dict_):
            H_out = cp.empty((len(X), self.n_components))
            for x, h_out in zip(X.to_arrow(), H_out): ## from cupy back to cudf
                h_out[:] = cp.asarray(self.H_dict_[x])
        elif self.engine != 'cuml':
            H_out = np.empty((len(X), self.n_components))

            for x, h_out in zip(X, H_out):
                try:
                    self.H_dict_[x]=self.H_dict_[x].get()
                except:
                    pass
                h_out[:] = self.H_dict_[x]
        else:
            H_out = cp.empty((len(X), self.n_components))
            for x, h_out in zip(X, H_out): ## from cupy back to cudf
                h_out[:] = self.H_dict_[x]
        return H_out

    def _init_w(self, V: NDArray, X) -> tuple[NDArray, NDArray, NDArray]:
        """
        Initialize the topics `W`.
        If `self.init='k-means++'`, we use the init method of
        sklearn.cluster.KMeans.
        If `self.init='random'`, topics are initialized with a Gamma
        distribution.
        If `self.init='k-means'`, topics are initialized with a KMeans on the
        n-grams counts.
        """
        if self.init == "k-means++":
            W, _ = kmeans_plusplus(
                V,
                self.n_components,
                x_squared_norms=row_norms(V, squared=True),
                random_state=self.random_state,
                n_local_trials=None,
            )
            W = W + 0.1  # To avoid restricting topics to a few n-grams only
        elif self.init == "random":
            W = self.random_state.gamma(
                shape=self.gamma_shape_prior,
                scale=self.gamma_scale_prior,
                size=(self.n_components, self.n_vocab),
            )
        elif self.init == "k-means":
            prototypes = get_kmeans_prototypes(
                X,
                self.n_components,
                analyzer=self.analyzer,
                random_state=self.random_state,
            )
            W = self.ngrams_count_.transform(prototypes).A + 0.1
            if self.add_words:
                W2 = self.word_count_.transform(prototypes).A + 0.1
                W = np.hstack((W, W2))
            # if k-means doesn't find the exact number of prototypes
            if W.shape[0] < self.n_components:
                W2, _ = kmeans_plusplus(
                    V,
                    self.n_components - W.shape[0],
                    x_squared_norms=row_norms(V, squared=True),
                    random_state=self.random_state,
                    n_local_trials=None,
                )
                W2 = W2 + 0.1
                W = np.concatenate((W, W2), axis=0)
        else:
            raise ValueError(f"Initialization method {self.init!r} does not exist. ")
        W = cp.array(W)
        W /= W.sum(axis=1, keepdims=True)
        A = cp.ones((self.n_components, self.n_vocab)) * 1e-10
        B = A.copy()
        return W, A, B

    def _minibatch_convergence(
        self,
        batch_size: int,
        batch_cost: float,
        n_samples: int,
        step: int,
        n_steps: int,
    ):
        """
        Helper function to encapsulate the early stopping logic.

        Parameters
        ----------
        batch_size : int
            The size of the current batch.
        batch_cost : float
            The cost (KL score) of the current batch.
        n_samples : int
            The total number of samples in X.
        step : int
            The current step (for verbose mode).
        n_steps : int
            The total number of steps (for verbose mode).

        Returns
        -------
        bool
            Whether the algorithm should stop or not.
        """
        # adapted from sklearn.decomposition.MiniBatchNMF

        # counts steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because H is not updated yet.
        if step == 1:
            if self.verbose:
                print(f"Minibatch step {step}/{n_steps}: mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_cost is None:
            self._ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_cost = self._ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch cost: "
                f"{batch_cost}, ewa cost: {self._ewa_cost}"
            )

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        if self._ewa_cost_min is None or self._ewa_cost < self._ewa_cost_min:
            self._no_improvement = 0
            self._ewa_cost_min = self._ewa_cost
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in objective function) "
                    f"at step {step}/{n_steps}"
                )
            return True

        return False

    def fit(self, X: ArrayLike, y=None) -> "GapEncoderColumn":
        """
        Fit the GapEncoder on `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance (self).
        """
        t = time()
        # Copy parameter rho
        self.rho_ = self.rho
<<<<<<<< HEAD:skrub/_gap_encoder.py
        # Attributes to monitor the convergence
        self._ewa_cost = None
        self._ewa_cost_min = None
        self._no_improvement = 0
========
        
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        # Check if first item has str or np.str_ type
        
        self.Xt_= df_type(X)
        # if 'series' not in self.Xt_ and X.shape[1]>1:
        # assert isinstance(X, str), "Input data is not string. "
        # else:
        #     assert isinstance(X.str.lower(), str), "Input data is not string. "
        # Make n-grams counts matrix unq_V
        unq_X, unq_V, lookup = self._init_vars(X)
        n_batch = (len(X) - 1) // self.batch_size + 1
<<<<<<<< HEAD:skrub/_gap_encoder.py
        n_samples = len(X)
        del X
========
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        # Get activations unq_H
        del X
        unq_H = self._get_H(unq_X)
<<<<<<<< HEAD:skrub/_gap_encoder.py
        converged = False
        for n_iter_ in range(self.max_iter):
            # Loop over batches
            for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                # Update activations unq_H
                unq_H[unq_idx] = _multiplicative_update_h(
                    unq_V[unq_idx],
========
        unq_V=csr_gpu(unq_V);unq_H=cp.array(unq_H); ## redundant

        for n_iter_ in range(self.max_iter):
            if (sys.getsizeof(unq_V)*sys.getsizeof(unq_V))<self.gmem and self.engine=='cuml':
                logger.debug(f"fitting smallfast-wise")
                W_type = df_type(self.W_)
                if 'cudf' not in W_type and 'cupy' not in W_type:
                    logger.debug(f"moving to gpu")
                    self.W_=cp.array(self.W_);self.B_=cp.array(self.B_);self.A_=cp.array(self.A_)
                elif 'cudf' in W_type:
                    logger.debug(f"keeping on gpu via cupy")
                    self.W_=self.W_.to_cupy();self.B_=self.B_.to_cupy();self.A_=self.A_.to_cupy()
                W_last = self.W_.copy()
                unq_H = _multiplicative_update_h_smallfast(
                    unq_V,
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
                    self.W_,
                    unq_H,
                    epsilon=1e-3,
                    max_iter=self.max_iter_e_step,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
                _multiplicative_update_w_smallfast(
                    unq_V,
                    self.W_,
                    self.A_,
                    self.B_,
                    unq_H,
                    self.rescale_W,
                    self.rho_,
                )
<<<<<<<< HEAD:skrub/_gap_encoder.py
                batch_cost = _beta_divergence(
                    unq_V[idx],
                    unq_H[idx],
                    self.W_,
                    "kullback-leibler",
                    square_root=False,
                ) / len(idx)
                if self._minibatch_convergence(
                    batch_size=len(idx),
                    batch_cost=batch_cost,
                    n_samples=n_samples,
                    step=n_iter_ * n_batch + i,
                    n_steps=self.max_iter * n_batch,
                ):
                    converged = True
                    break
            if converged:
                break

        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
========
            else:
                W_type = df_type(self.W_)
                if self.engine=='cuml' and (sys.getsizeof(unq_V)*sys.getsizeof(unq_V))>self.gmem:
                    if 'cudf' not in W_type and 'cupy' not in W_type:
                        self.W_=cp.array(self.W_);self.B_=cp.array(self.B_);self.A_=cp.array(self.A_);
                        logger.debug(f"moving to cupy")
                    elif 'cudf' in W_type:
                        self.W_=self.W_.to_cupy();self.B_=self.B_.to_cupy();self.A_=self.A_.to_cupy();
                        logger.debug(f"keeping on gpu via cupy")
                    # Loop over batches
                else:
                    
                    if hasattr(unq_H, 'device') or 'cupy' in W_type:
                        # unq_V=unq_V.get();unq_H=unq_H.get();
                        logger.debug(f"force numpy fit")
                for i, (unq_idx, idx) in enumerate(batch_lookup(lookup, n=self.batch_size)):
                    if i == n_batch - 1:
                        W_last = self.W_.copy()
                    # Update activations unq_H
                    unq_H[unq_idx] = _multiplicative_update_h(
                        self,
                        unq_V[unq_idx],
                        self.W_,
                        unq_H[unq_idx],
                        epsilon=1e-3,
                        max_iter=self.max_iter_e_step,
                        rescale_W=self.rescale_W,
                        gamma_shape_prior=self.gamma_shape_prior,
                        gamma_scale_prior=self.gamma_scale_prior,
                    )
                    # Update the topics self.W_
                    _multiplicative_update_w(
                        self,
                        unq_V[idx],
                        self.W_,
                        self.A_,
                        self.B_,
                        unq_H[idx],
                        self.rescale_W,
                        self.rho_,
                    )

            # if i == n_batch - 1:
                # Compute the norm of the update of W in the last batch
            W_change = cp.multiply(cp.linalg.norm(self.W_ - W_last), cp.reciprocal(cp.linalg.norm(W_last)))

            if (W_change < self.tol) and (n_iter_ >= self.min_iter - 1):
                break  # Stop if the change in W is smaller than the tolerance
        if 'cudf' in df_type(unq_X) :
            self.H_dict_.update(zip(unq_X.to_arrow(), unq_H))
        else:
            self.H_dict_.update(zip(unq_X, unq_H))
        logger.debug(
            f"--GapEncoder Fitting took {(time() - t) / 60:.2f} minutes\n"
        )
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        return self

    def get_feature_names_out(
        self,
        n_labels: int = 3,
        prefix: str = "",
    ) -> list[str]:
        """
        Returns the labels that best summarize the learned components/topics.
        For each topic, labels with the highest activations are selected.

        Parameters
        ----------
        n_labels : int, default=3
            The number of labels used to describe each topic.
        prefix : str, default=''
            Used as a prefix for the categories.

        Returns
        -------
        list of str
            The labels that best describe each topic.
        """

<<<<<<<< HEAD:skrub/_gap_encoder.py
        vectorizer = CountVectorizer()
        vectorizer.fit(list(self.H_dict_.keys()))
        vocabulary = np.array(vectorizer.get_feature_names_out())
        encoding = self.transform(np.array(vocabulary).reshape(-1))
        encoding = abs(encoding)
========
        vectorizer = self._CV()#lowercase=False,preprocessor=None)

        if 'cudf'  in self.Xt_:
            A=cudf.Series([(item).as_py() for item in self.H_dict_.keys()])
            vectorizer.fit(A)
            vocabulary = (vectorizer.get_feature_names().to_arrow())
            encoding = self.transform(cudf.Series(vocabulary))
            encoding = abs(encoding)
            vocabulary = (cp.asnumpy(vocabulary).reshape(-1)) ## after fit to build labels below
            encoding = (cp.asnumpy(encoding))

        else:
            vectorizer.fit(list(self.H_dict_.keys()))
            vocabulary = np.array(vectorizer.get_feature_names())
            encoding = self.transform(pd.Series(vocabulary).reshape(-1))
            encoding = abs(encoding)

>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        encoding = encoding / np.sum(encoding, axis=1, keepdims=True)
        n_components = encoding.shape[1]
        topic_labels = []
        for i in range(n_components):
            x = encoding[:, i]
            labels = vocabulary[np.argsort(-x)[:n_labels]]
            topic_labels.append(labels)

        topic_labels = [prefix + ", ".join(label) for label in topic_labels]
        return topic_labels

<<<<<<<< HEAD:skrub/_gap_encoder.py
    def score(self, X: ArrayLike) -> float:
        """Score this instance of `X`.

        Returns the Kullback-Leibler divergence between the n-grams counts
        matrix `V` of `X`, and its non-negative factorization `HW`.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The data to encode.

        Returns
        -------
        float
            The Kullback-Leibler divergence.
        """

        # Build n-grams/word counts matrix
        unq_X, lookup = np.unique(X, return_inverse=True)
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

        self._add_unseen_keys_to_H_dict(unq_X)
        unq_H = self._get_H(unq_X)
        # Given the learnt topics W, optimize the activations H to fit V = HW
        for slice in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            unq_H[slice] = _multiplicative_update_h(
                unq_V[slice],
                self.W_,
                unq_H[slice],
                epsilon=1e-3,
                max_iter=self.max_iter_e_step,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        # Compute the KL divergence between V and HW
        kl_divergence = _beta_divergence(
            unq_V[lookup], unq_H[lookup], self.W_, "kullback-leibler", square_root=False
        )
        return kl_divergence

    def partial_fit(self, X: ArrayLike, y=None) -> "GapEncoderColumn":
        """Partial fit this instance on `X`.

        To be used in an online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, )
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoderColumn
            The fitted GapEncoderColumn instance (self).
        """

        # Init H_dict_ with empty dict if it's the first call of partial_fit
        if not hasattr(self, "H_dict_"):
            self.H_dict_ = dict()
        # Same thing for the rho_ parameter
        if not hasattr(self, "rho_"):
            self.rho_ = self.rho
        # Check if first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        # Check if it is not the first batch
        if hasattr(self, "vocabulary"):  # Update unq_X, unq_V with new batch
            unq_X, lookup = np.unique(X, return_inverse=True)
            unq_V = self.ngrams_count_.transform(unq_X)
            if self.add_words:
                unq_V2 = self.word_count_.transform(unq_X)
                unq_V = sparse.hstack((unq_V, unq_V2), format="csr")

            unseen_X = np.setdiff1d(unq_X, np.array([*self.H_dict_]))
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            if unseen_V.shape[0] != 0:
                unseen_H = _rescale_h(
                    unseen_V, np.ones((len(unseen_X), self.n_components))
                )
                for x, h in zip(unseen_X, unseen_H):
                    self.H_dict_[x] = h
                del unseen_H
            del unseen_X, unseen_V
        else:  # If it is the first batch, call _init_vars to init unq_X, unq_V
            unq_X, unq_V, lookup = self._init_vars(X)

        unq_H = self._get_H(unq_X)
        # Update unq_H, the activations
        unq_H = _multiplicative_update_h(
            unq_V,
            self.W_,
            unq_H,
            epsilon=1e-3,
            max_iter=self.max_iter_e_step,
            rescale_W=self.rescale_W,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
        )
        # Update the topics self.W_
        _multiplicative_update_w(
            unq_V[lookup],
            self.W_,
            self.A_,
            self.B_,
            unq_H[lookup],
            self.rescale_W,
            self.rho_,
        )
        # Update self.H_dict_ with the learned encoded vectors (activations)
        self.H_dict_.update(zip(unq_X, unq_H))
        return self
========
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py

    def _add_unseen_keys_to_H_dict(self, X) -> None:
        """
        Add activations of unseen string categories from `X` to `H_dict`.
        """
        
        # def setdiff1d(ar1, ar2, assume_unique=False):
        #     if assume_unique:
        #         ar1 = cp.ravel(ar1)
        #     else:
        #         ar1 = cudf.Series(ar1).unique
        #         ar2 = cudf.Series(ar2).unique
        #     return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
    
        if 'cudf' in self.Xt_:
            A=np.array([(item).as_py() for item in self.H_dict_])
            unseen_X = np.setdiff1d(X.to_arrow(), A, assume_unique=True) 
            unseen_X = cudf.Series(unseen_X)
        else:
            unseen_X = np.setdiff1d(X, np.array([*self.H_dict_]))
        if unseen_X.size > 0:
            unseen_V = self.ngrams_count_.transform(unseen_X)
            if self.add_words:
                unseen_V2 = self.word_count_.transform(unseen_X)
                unseen_V = sparse.hstack((unseen_V, unseen_V2), format="csr")

            unseen_H = _rescale_h(self,unseen_V, np.ones((unseen_V.shape[0], self.n_components)))
            
            if 'cudf' in df_type(unseen_X) :
                self.H_dict_.update(zip(unseen_X.to_arrow(), unseen_H.values))
            else:
                self.H_dict_.update(zip(unseen_X, unseen_H))

    def transform(self, X: ArrayLike) -> NDArray:
        """Return the encoded vectors (activations) `H` of input strings in `X`.

        Parameters
        ----------
        X : array-like, shape (n_samples)
            The string data to encode.

        Returns
        -------
        ndarray, shape (n_samples, n_topics)
            Transformed input.
        """
        t = time()
        check_is_fitted(self, "H_dict_")
<<<<<<<< HEAD:skrub/_gap_encoder.py
        # Copy the state of H before continuing fitting it
        pre_trans_H_dict_ = deepcopy(self.H_dict_)
        # Check if the first item has str or np.str_ type
        assert isinstance(X[0], str), "Input data is not string. "
        unq_X = np.unique(X)
========
        # Check if first item has str or np.str_ type
        # assert isinstance(X[0], str), "Input data is not string. "
        unq_X = X.unique() # np.unique(X)

>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        # Build the n-grams counts matrix V for the string data to encode
        unq_V = self.ngrams_count_.transform(unq_X)
        if self.add_words:  # Add words counts
            unq_V2 = self.word_count_.transform(unq_X)
            unq_V = sparse.hstack((unq_V, unq_V2), format="csr")
        # Add unseen strings in X to H_dict
        self._add_unseen_keys_to_H_dict(unq_X) ### need to get this back for transforming obviously
        unq_H = self._get_H(unq_X)
        # Loop over batches
<<<<<<<< HEAD:skrub/_gap_encoder.py
        for slc in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
            # Given the learnt topics W, optimize H to fit V = HW
            unq_H[slc] = _multiplicative_update_h(
                unq_V[slc],
                self.W_,
                unq_H[slc],
                epsilon=1e-3,
                max_iter=100,
                rescale_W=self.rescale_W,
                gamma_shape_prior=self.gamma_shape_prior,
                gamma_scale_prior=self.gamma_scale_prior,
            )
        # Store and return the encoded vectors of X
        self.H_dict_.update(zip(unq_X, unq_H))
        feature_names_out = self._get_H(X)
        # Restore H
        self.H_dict_ = pre_trans_H_dict_
        return feature_names_out
========
        logger.info(f"req gpu mem =  `{(sys.getsizeof(unq_V)*sys.getsizeof(unq_V))}`, ie `{unq_V.shape[0]*unq_V.shape[1]}`")
        if self.engine=='cuml' and (sys.getsizeof(unq_V)*sys.getsizeof(unq_V))<self.gmem:
            logger.debug(f"smallfast transform")
            unq_H = _multiplicative_update_h_smallfast(
                    unq_V,
                    self.W_,
                    unq_H,
                    epsilon=1e-3,
                    max_iter=100,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
        else:
            W_type = df_type(self.W_)
            if self.engine=='cuml' and (sys.getsizeof(unq_V)*sys.getsizeof(unq_V))>self.gmem:
                logger.debug(f"cupy transform")
                if 'cudf' not in W_type and 'cupy' not in W_type:
                    self.W_=cp.array(self.W_);unq_V=csr_gpu(unq_V);unq_H=cp.array(unq_H);
                    logger.debug(f"moving to cupy")
                if 'cudf' in W_type:
                    self.W_=self.W_.to_cupy(); unq_V=unq_V.to_cupy();unq_H=unq_H.to_cupy();
                    logger.debug(f"kept on gpu via cupy")
            else:
                # self.W_=self.W_.get();
                if hasattr(unq_H, 'device') or 'cupy' in W_type or self.engine !='cuml':
                    logger.debug(f"force numpy transform")
            for slc in gen_batches(n=unq_H.shape[0], batch_size=self.batch_size):
                # Given the learnt topics W, optimize H to fit V = HW
                unq_H[slc] = _multiplicative_update_h(
                    self,
                    unq_V[slc],
                    self.W_,
                    unq_H[slc],
                    epsilon=1e-3,
                    max_iter=100,
                    rescale_W=self.rescale_W,
                    gamma_shape_prior=self.gamma_shape_prior,
                    gamma_scale_prior=self.gamma_scale_prior,
                )
        if 'cudf' in df_type(unq_X) :
            self.H_dict_.update(zip(unq_X.to_arrow(), unq_H))
            # self._get_H(X)
        else:
            self.H_dict_.update(zip(unq_X, unq_H))
            # cudf.DataFrame.from_dict(self._get_H(X))
        logger.debug(
            f"--GapEncoder Tranforming took {(time() - t) / 60:.2f} minutes\n"
        )
        
        return self._get_H(X)
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py


class GapEncoder(TransformerMixin, BaseEstimator):
    """Constructs latent topics with continuous encoding.

    This encoder can be understood as a continuous encoding on a set of latent
    categories estimated from the data. The latent categories are built by
    capturing combinations of substrings that frequently co-occur.

<<<<<<<< HEAD:skrub/_gap_encoder.py
    The GapEncoder supports online learning on batches of
    data for scalability through the GapEncoder.partial_fit
========
    The :class:`~cu_cat.GapEncoder` supports online learning on batches of
    data for scalability through the :func:`~cu_cat.GapEncoder.partial_fit`
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    method.

    The principle is as follows:

    1. Given an input string array `X`, we build its bag-of-n-grams
       representation `V` (`n_samples`, `vocab_size`).
    2. Instead of using the n-grams counts as encodings, we look for low-
       dimensional representations by modeling n-grams counts as linear
       combinations of topics ``V = HW``, with `W` (`n_topics`, `vocab_size`)
       the topics and `H` (`n_samples`, `n_topics`) the associated activations.
    3. Assuming that n-grams counts follow a Poisson law, we fit `H` and `W` to
       maximize the likelihood of the data, with a Gamma prior for the
       activations `H` to induce sparsity.
    4. In practice, this is equivalent to a non-negative matrix factorization
       with the Kullback-Leibler divergence as loss, and a Gamma prior on `H`.
       We thus optimize `H` and `W` with the multiplicative update method.

    Parameters
    ----------
    n_components : int, optional, default=10
        Number of latent categories used to model string data.
    batch_size : int, optional, default=1024
        Number of samples per batch.
    gamma_shape_prior : float, optional, default=1.1
        Shape parameter for the Gamma prior distribution.
    gamma_scale_prior : float, optional, default=1.0
        Scale parameter for the Gamma prior distribution.
    rho : float, optional, default=0.95
        Weight parameter for the update of the `W` matrix.
    rescale_rho : bool, optional, default=False
        If `True`, use ``rho ** (batch_size / len(X))`` instead of rho to obtain
        an update rate per iteration that is independent of the batch size.
    hashing : bool, optional, default=False
        If `True`, HashingVectorizer is used instead of CountVectorizer.
        It has the advantage of being very low memory, scalable to large
        datasets as there is no need to store a vocabulary dictionary in
        memory.
    hashing_n_features : int, default=2**12
        Number of features for the HashingVectorizer.
        Only relevant if `hashing=True`.
    init : {'k-means++', 'random', 'k-means'}, default='k-means++'
        Initialization method of the `W` matrix.
        If `init='k-means++'`, we use the init method of KMeans.
        If `init='random'`, topics are initialized with a Gamma distribution.
        If `init='k-means'`, topics are initialized with a KMeans on the
        n-grams counts.
    max_iter : int, default=5
        Maximum number of iterations on the input data.
    ngram_range : int 2-tuple, default=(2, 4)
       The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    analyzer : {'word', 'char', 'char_wb'}, default='char'
        Analyzer parameter for the HashingVectorizer / CountVectorizer.
        Describes whether the matrix `V` to factorize should be made of
        word counts or character-level n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    add_words : bool, default=False
        If `True`, add the words counts to the bag-of-n-grams representation
        of the input data.
    random_state : int or RandomState, optional
        Random number generator seed for reproducible output across multiple
        function calls.
    rescale_W : bool, default=True
        If `True`, the weight matrix `W` is rescaled at each iteration
        to have a l1 norm equal to 1 for each row.
    max_iter_e_step : int, default=1
        Maximum number of iterations to adjust the activations h at each step.
    max_no_improvement : int, default=5
        Control early stopping based on the consecutive number of mini batches
        that do not yield an improvement on the smoothed cost function.
        To disable early stopping and run the process fully,
        set ``max_no_improvement=None``.
    handle_missing : {'error', 'empty_impute'}, default='empty_impute'
        Whether to raise an error or impute with empty string ('') if missing
        values (NaN) are present during GapEncoder.fit (default is to impute).
        In GapEncoder.inverse_transform, the missing categories will
        be denoted as `None`.
    n_jobs : int, optional
        The number of jobs to run in parallel.
        The process is parallelized column-wise,
        meaning each column is fitted in parallel. Thus, having
        `n_jobs` > X.shape[1] will not speed up the computation.
    verbose : int, default=0
        Verbosity level. The higher, the more granular the logging.

    Attributes
    ----------
    rho_ : float
        Effective update rate for the `W` matrix.
    fitted_models_ : list of GapEncoderColumn
        Column-wise fitted GapEncoders.
    column_names_ : list of str
        Column names of the data the Gap was fitted on.

    See Also
    --------
<<<<<<<< HEAD:skrub/_gap_encoder.py
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.
    deduplicate :
========
    :class:`~cu_cat.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`~cu_cat.SimilarityEncoder` :
        Encode string columns as a numeric array with n-gram string similarity.
    :class:`~cu_cat.deduplicate` :
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        Deduplicate data by hierarchically clustering similar strings.

    References
    ----------
    For a detailed description of the method, see
    `Encoding high-cardinality string categorical variables
    <https://hal.inria.fr/hal-02171256v4>`_ by Cerda, Varoquaux (2019).

    Examples
    --------
    >>> enc = GapEncoder(n_components=2)

    Let's encode the following non-normalized data:

    >>> X = [['paris, FR'], ['Paris'], ['London, UK'], ['Paris, France'],
             ['london'], ['London, England'], ['London'], ['Pqris']]

    >>> enc.fit(X)
    GapEncoder(n_components=2)

<<<<<<<< HEAD:skrub/_gap_encoder.py
    The GapEncoder has found the following two topics:
========
    The :class:`~cu_cat.GapEncoder` has found the following two topics:
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py

    >>> enc.get_feature_names_out()
    ['england, london, uk', 'france, paris, pqris']

    It got it right, reccuring topics are "London" and "England" on the
    one side and "Paris" and "France" on the other.

    As this is a continuous encoding, we can look at the level of
    activation of each topic for each category:

    >>> enc.transform(X)
    array([[ 0.05202843, 10.54797156],
          [ 0.05000118,  4.54999882],
          [12.04734788,  0.05265212],
          [ 0.05263068, 16.54736932],
          [ 6.04999624,  0.05000376],
          [19.546716  ,  0.053284  ],
          [ 6.04999623,  0.05000376],
          [ 0.05002016,  4.54997983]])

    The higher the value, the bigger the correspondance with the topic.
    """

    rho_: float
    fitted_models_: list[GapEncoderColumn]
    column_names_: list[str]

    def __init__(
        self,
        *,
        n_components: int = 10,
        batch_size: int = 1024,
        gamma_shape_prior: float = 1.1,
        gamma_scale_prior: float = 1.0,
        rho: float = 0.95,
        rescale_rho: bool = False,
        hashing: bool = False,
        hashing_n_features: int = 2**12,
<<<<<<<< HEAD:skrub/_gap_encoder.py
        init: Literal["k-means++", "random", "k-means"] = "k-means++",
========
        init: Literal["k-means++", "random", "k-means"] = "random",
        tol: float = 1e-4,
        min_iter: int = 2,
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        max_iter: int = 5,
        ngram_range: tuple[int, int] = (2, 4),
        analyzer: Literal["word", "char", "char_wb"] = "char",
        add_words: bool = False,
        random_state: int | RandomState | None = None,
        rescale_W: bool = True,
        max_iter_e_step: int = 1,
        max_no_improvement: int = 5,
        handle_missing: Literal["error", "empty_impute"] = "zero_impute",
<<<<<<<< HEAD:skrub/_gap_encoder.py
        n_jobs: int | None = None,
        verbose: int = 0,
========
        engine: Engine = "auto",

>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    ):
        engine_resolved = resolve_engine(engine)
        # FIXME remove as set_new_kwargs will always replace?
        if engine_resolved == 'sklearn':
            _, _, engine = lazy_sklearn_import_has_dependancy()
            from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
        elif engine_resolved == 'cuml':
            _, _, engine, gmem = lazy_cuml_import_has_dependancy()
            from cuml.feature_extraction.text import CountVectorizer,HashingVectorizer
        
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.gamma_shape_prior = gamma_shape_prior  # 'a' parameter
        self.gamma_scale_prior = gamma_scale_prior  # 'b' parameter
        self.rho = rho
        self.rescale_rho = rescale_rho
        self.batch_size = batch_size
        self.hashing = hashing
        self.hashing_n_features = hashing_n_features
        self.max_iter = max_iter
        self.init = init
        self.analyzer = analyzer
        self.add_words = add_words
        self.random_state = random_state
        self.rescale_W = rescale_W
        self.max_iter_e_step = max_iter_e_step
        self.max_no_improvement = max_no_improvement
        self.handle_missing = handle_missing
<<<<<<<< HEAD:skrub/_gap_encoder.py
        self.n_jobs = n_jobs
        self.verbose = verbose
========
        self._CV = CountVectorizer
        self._HV = HashingVectorizer
        self.engine = engine_resolved
        self.gmem = gmem[0]


    def _more_tags(self) -> Dict[str, List[str]]:
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py

    def _create_column_gap_encoder(self) -> GapEncoderColumn:
        """Helper method for creating a GapEncoderColumn from
        the parameters of this instance."""
        return GapEncoderColumn(
            ngram_range=self.ngram_range,
            n_components=self.n_components,
            analyzer=self.analyzer,
            gamma_shape_prior=self.gamma_shape_prior,
            gamma_scale_prior=self.gamma_scale_prior,
            rho=self.rho,
            rescale_rho=self.rescale_rho,
            batch_size=self.batch_size,
            hashing=self.hashing,
            hashing_n_features=self.hashing_n_features,
            max_iter=self.max_iter,
            init=self.init,
            add_words=self.add_words,
            random_state=self.random_state,
            rescale_W=self.rescale_W,
            max_iter_e_step=self.max_iter_e_step,
<<<<<<<< HEAD:skrub/_gap_encoder.py
            max_no_improvement=self.max_no_improvement,
            verbose=self.verbose,
========
            engine=self.engine
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        )

    def _handle_missing(self, X):
        """
        Imputes missing values with `` or raises an error
        Note: modifies the array in-place.
        """
        if self.handle_missing not in ["error", "zero_impute"]:
            raise ValueError(
                "handle_missing should be either 'error' or "
                f"'zero_impute', got {self.handle_missing!r}. "
            )
        if 'cudf' not in self.Xt_:
            missing_mask = _object_dtype_isnan(X)
        
            if missing_mask.any():
                if self.handle_missing == "error":
                    raise ValueError("Input data contains missing values. ")
                elif self.handle_missing == "zero_impute":
                    X[missing_mask] = ""
        else:
            missing_mask = _object_dtype_isnan(X.to_pandas())
            if sum(missing_mask) != 0:
                if self.handle_missing == "error":
                    raise ValueError("Input data contains missing values. ")
                elif self.handle_missing == "zero_impute":
                    X[missing_mask] = ""
        return X

    def fit(self, X: ArrayLike, y=None) -> "GapEncoder":
        """Fit the instance on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
<<<<<<<< HEAD:skrub/_gap_encoder.py
        GapEncoder
            The fitted GapEncoder instance (self).
========
        :class:`~cu_cat.GapEncoder`
            Fitted :class:`~cu_cat.GapEncoder` instance (self).
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        """

        # Check that n_samples >= n_components
        n_samples = _num_samples(X)
        if n_samples < self.n_components:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_components={self.n_components}. "
            )
        # Copy parameter rho
        self.rho_ = self.rho
        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
<<<<<<<< HEAD:skrub/_gap_encoder.py
        X = check_input(X)
        self._check_n_features(X, reset=True)
        X = self._handle_missing(X)
        self.fitted_models_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._create_column_gap_encoder().fit)(X[:, k])
            for k in range(X.shape[1])
        )
========
        self.Xt_ = df_type(X)
        if 'cudf' not in self.Xt_ or 'cuml' != self.engine:
            X = check_input(X)
            try:
                X = X.to_pandas()
            except:
                pass
            # X = self._handle_missing(X)
            self.fitted_models_ = []
            
            for k in range(X.shape[1]):
                col_enc = self._create_column_gap_encoder()
                self.fitted_models_.append(col_enc.fit(X.iloc[:, k]))
        else :
            X = check_input(X)
            # X = self._handle_missing(X)
            self.fitted_models_ = []
            for k in range(X.shape[1]):
                col_enc = self._create_column_gap_encoder()
                self.fitted_models_.append(col_enc.fit(X.iloc[:,k]))
            
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        return self

    def transform(self, X: ArrayLike) -> NDArray:
        """Return the encoded vectors (activations) `H` of input strings in `X`.

        Given the learnt topics `W`, the activations `H` are tuned to fit
        ``V = HW``. When `X` has several columns, they are encoded separately
        and then concatenated.

        Remark: calling transform multiple times in a row on the same
        input `X` can give slightly different encodings. This is expected
        due to a caching mechanism to speed things up.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to encode.

        Returns
        -------
        ndarray, shape (n_samples, n_topics * n_features)
            Transformed input.
        """
        check_is_fitted(self, "fitted_models_")
        # Check input data shape
        X = check_input(X)
<<<<<<<< HEAD:skrub/_gap_encoder.py
        self._check_n_features(X, reset=False)
        X = self._handle_missing(X)
========
        
        # X = self._handle_missing(X)
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
        X_enc = []
        if 'cudf' in str(getmodule(X)) or 'cuml' == self.engine:
            for k in range(X.shape[1]):
                X_enc.append(self.fitted_models_[k].transform(X.iloc[:, k]))
        else:
            for k in range(X.shape[1]):
                X_enc.append(self.fitted_models_[k].transform(X.iloc[:, k]))
        X_enc = np.hstack(X_enc)
        return X_enc

<<<<<<<< HEAD:skrub/_gap_encoder.py
    def partial_fit(self, X: ArrayLike, y=None) -> "GapEncoder":
        """Partial fit this instance on X.

        To be used in an online learning procedure where batches of data are
        coming one by one.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The string data to fit the model on.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        GapEncoder
            The fitted GapEncoder instance (self).
        """

        # If X is a dataframe, store its column names
        if isinstance(X, pd.DataFrame):
            self.column_names_ = list(X.columns)
        # Check input data shape
        X = check_input(X)
        X = self._handle_missing(X)
        # Init the `GapEncoderColumn` instances if the model was
        # not fitted already.
        if not hasattr(self, "fitted_models_"):
            self.fitted_models_ = [
                self._create_column_gap_encoder() for _ in range(X.shape[1])
            ]
        for k in range(X.shape[1]):
            self.fitted_models_[k].partial_fit(X[:, k])
        return self

========
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    def get_feature_names_out(
        self,
        col_names: Literal["auto"] | list[str] | None = None,
        n_labels: int = 3,
        input_features=None,
    ):
        """Return the labels that best summarize the learned components/topics.

        For each topic, labels with the highest activations are selected.

        Parameters
        ----------
        col_names : 'auto' or list of str, optional
            The column names to be added as prefixes before the labels.
            If `col_names=None`, no prefixes are used.
            If `col_names='auto'`, column names are automatically defined:

                - if the input data was a :obj:`~pandas.DataFrame`,
                  its column names are used,
                - otherwise, 'col1', ..., 'colN' are used as prefixes.

            Prefixes can be manually set by passing a list for `col_names`.

        n_labels : int, default=3
            The number of labels used to describe each topic.

        input_features : None
            Unused, only here for compatibility.

        Returns
        -------
        list of str
            The labels that best describe each topic.
            Each element contains the labels joined by a comma.
        """
        check_is_fitted(self, "fitted_models_")
        # Generate prefixes
        if isinstance(col_names, str) and col_names == "auto":
            if hasattr(self, "column_names_"):  # Use column names
                prefixes = ["%s: " % col for col in self.column_names_]
            else:  # Use 'col1: ', ... 'colN: ' as prefixes
                prefixes = ["col%d: " % i for i in range(len(self.fitted_models_))]
        elif col_names is None:  # Empty prefixes
            prefixes = [""] * len(self.fitted_models_)
        else:
            prefixes = ["%s: " % col for col in col_names]
        labels = list()
        for k, enc in enumerate(self.fitted_models_):
            col_labels = enc.get_feature_names_out(n_labels, prefixes[k])
            labels.extend(col_labels)
        return labels

<<<<<<<< HEAD:skrub/_gap_encoder.py
    def score(self, X: ArrayLike, y=None) -> float:
        """Score this instance on `X`.

        Returns the sum over the columns of `X` of the Kullback-Leibler
        divergence between the n-grams counts matrix `V` of `X`, and its
        non-negative factorization `HW`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to encode.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        float
            The Kullback-Leibler divergence.
        """
        X = check_input(X)
        kl_divergence = 0
        for k in range(X.shape[1]):
            kl_divergence += self.fitted_models_[k].score(X[:, k])
        return kl_divergence

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical", "string"],
            "preserves_dtype": [],
            "allow_nan": True,
            "_xfail_checks": {
                "check_transformer_n_iter": "Don't want to add an `n_iter_` attribute.",
                "check_estimator_sparse_data": (
                    "Cannot create sparse matrix with strings."
                ),
                "check_estimators_dtypes": "We only support string dtypes.",
            },
        }


def _rescale_W(W: NDArray, A: NDArray) -> None:
========
    def get_feature_names(
        self, input_features=None, col_names: List[str] = None, n_labels: int = 3
    ) -> List[str]:
    #     """
    #     Ensures compatibility with sklearn < 1.0.
    #     Use `get_feature_names_out` instead.
    #     """
    #     # if parse_version(sklearn_version) >= parse_version("1.0"):
    #     #     warnings.warn(
    #     #         "Following the changes in scikit-learn 1.0, "
    #     #         "get_feature_names is deprecated. "
    #     #         "Use get_feature_names_out instead. ",
    #     #         DeprecationWarning,
    #     #         stacklevel=2,
    #     #     )
        return self.get_feature_names_out(col_names, n_labels)

def _rescale_W(W: np.array, A: np.array) -> None:
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    """
    Rescale the topics `W` to have a L1-norm equal to 1.
    Note that they are modified in-place.
    """
    s = W.sum(axis=1, keepdims=True)
    W /= s
    A /= s


def _special_sparse_dot(H, W, X):
    """Computes np.dot(H, W), only where X is non zero."""
    # adapted from sklearn.decomposition.MiniBatchNMF
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = H.shape[1]
        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(H[ii[batch], :], W.T[jj[batch], :]).sum(
                axis=1
            )

        HW = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        # in sklearn, it was return WH.tocsr(), but it breaks the code in our case
        # I'm not sure why
        return HW
    else:
        return np.dot(H, W)


def _multiplicative_update_w(
<<<<<<<< HEAD:skrub/_gap_encoder.py
    Vt: NDArray,
    W: NDArray,
    A: NDArray,
    B: NDArray,
    Ht: NDArray,
========
    self,
    Vt: np.array,
    W: np.array,
    A: np.array,
    B: np.array,
    Ht: np.array,
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    rescale_W: bool,
    rho: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Multiplicative update step for the topics `W`.
    """
    if 'cudf' in df_type(Vt) or 'cupy' in df_type(Vt):
        A *= rho
        A += cp.multiply(W, safe_sparse_dot(Ht.T, Vt.multiply(1 / (cp.dot(Ht, W) + 1e-10))))
        B *= rho
        B += Ht.sum(axis=0).reshape(-1, 1)
        W=cp.multiply(A, cp.reciprocal(B))#, out=W)
        if rescale_W:
            _rescale_W(W, A)
        cp._default_memory_pool.free_all_blocks()
        
    else: # self.engine!='cuml':
        try:
            W=W.get()
        except:
            pass
        try:
            Ht=Ht.get()
        except:
            pass
        try:
            Vt=Vt.get()
        except:
            pass
        try:
            A=A.get()
        except:
            pass
        try:
            B=B.get()
        except:
            pass
        A *= rho
        A += np.multiply(W, safe_sparse_dot(Ht.T, Vt.multiply(1 / (np.dot(Ht, W) + 1e-10))))
        B *= rho
        B += Ht.sum(axis=0).reshape(-1, 1)
        W=np.multiply(A, np.reciprocal(B))#, out=W)
        if rescale_W:
            _rescale_W(W, A)
    return W,A,B #cp.array(W), cp.array(A), cp.array(B)

def _multiplicative_update_w_smallfast(
    Vt: np.array,
    W: np.array,
    A: np.array,
    B: np.array,
    Ht: np.array,
    rescale_W: bool,
    rho: float,
) -> Tuple[np.array, np.array, np.array]:
    """
    Multiplicative update step for the topics W.
    """
    # A=cp.array(A)
    # B=cp.array(B)
    # Ht=cp.array(Ht) ## not needed if we figure way to .get() outside function
    # W=cp.array(W)
    # Vt=csr_gpu(Vt)
    A *= rho
<<<<<<<< HEAD:skrub/_gap_encoder.py
    HtW = _special_sparse_dot(Ht, W, Vt)
    Vt_data = Vt.data
    HtW_data = HtW.data
    np.divide(Vt_data, HtW_data + 1e-10, out=Vt_data)
    HtVt = safe_sparse_dot(Ht.T, Vt)
    A += W * HtVt
========
    C = cp.matmul(Ht, W)
    R = Vt.multiply(cp.reciprocal(C) + 1e-10)
    T = R.T.dot(Ht).T ## drop .T on sparse Vt
    A += cp.multiply(W, T)
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    B *= rho
    B += Ht.sum(axis=0).reshape(-1, 1)
    cp.multiply(A, 1/B, out=W)
    if rescale_W:
        _rescale_W(W, A)
    # W=W.get();A=A.get();B=B.get()
    del C,R,T,Ht,Vt
    cp._default_memory_pool.free_all_blocks()
    return W,A,B

<<<<<<<< HEAD:skrub/_gap_encoder.py

def _rescale_h(V: NDArray, H: NDArray) -> NDArray:
========
def _rescale_h(self, V: np.array, H: np.array) -> np.array:
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    """
    Rescale the activations `H`.
    """
    epsilon = 1e-10  # in case of a document having length=0
    # if self.engine=='cuml':
    if 'cupy' in df_type(V):
        H = cp.array(H)
        H *= cp.maximum(epsilon, V.sum(axis=1))
    else:
        H *= np.maximum(epsilon, V.sum(axis=1).A)
    H /= H.sum(axis=1, keepdims=True)

    return cudf.DataFrame(H)


def _multiplicative_update_h(
<<<<<<<< HEAD:skrub/_gap_encoder.py
    Vt: NDArray,
    W: NDArray,
    Ht: NDArray,
========
    self,
    Vt: np.array,
    W: np.array,
    Ht: np.array,
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    epsilon: float = 1e-3,
    max_iter: int = 10,
    rescale_W: bool = False,
    gamma_shape_prior: float = 1.1,
    gamma_scale_prior: float = 1.0,
):
    """
    Multiplicative update step for the activations `H`.
    """
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    const = (gamma_shape_prior - 1) / WT1
<<<<<<<< HEAD:skrub/_gap_encoder.py
    squared_epsilon = epsilon**2
    for vt, ht in zip(Vt, Ht):
        vt_ = vt.data
        idx = vt.indices
        W_WT1_ = W_WT1[:, idx]
        W_ = W[:, idx]
        squared_norm = 1
        for _ in range(max_iter):
            if squared_norm <= squared_epsilon:
                break
            aux = np.dot(W_WT1_, vt_ / (np.dot(ht, W_) + 1e-10))
            ht_out = ht * aux + const
            squared_norm = np.dot(ht_out - ht, ht_out - ht) / np.dot(ht, ht)
            ht[:] = ht_out
========
    squared_epsilon = epsilon #**2
    
    if 'cudf' in df_type(Vt) or 'cupy' in df_type(Vt):
        for vt, ht in zip(Vt, Ht):
            vt_ = vt.data
            idx = vt.indices
            W_WT1_ = W_WT1[:, idx]
            W_ = W[:, idx]
            squared_norm = 1
            for n_iter_ in range(max_iter):
                if squared_norm <= squared_epsilon:
                    break
                aux = cp.dot(W_WT1_, cp.multiply(vt_,cp.reciprocal(cp.dot(ht, W_) + 1e-10)))
                ht_out = cp.multiply(ht, aux) + const
                squared_norm = cp.multiply(cp.dot(ht_out - ht, ht_out - ht), cp.reciprocal(cp.dot(ht, ht)))
                ht[:] = ht_out
        del Vt,W_,W_WT1,ht,ht_out,vt,vt_
        cp._default_memory_pool.free_all_blocks()
    else:
        
        # for vt, ht in zip(Vt, Ht):
        #     vt_ = vt.data
        #     idx = vt.indices
        #     W_WT1_ = W_WT1[:, idx]
        #     W_ = W[:, idx]
        #     squared_norm = 1
        #     for n_iter_ in range(max_iter):
        #         if squared_norm <= squared_epsilon:
        #             break
        #         aux = np.dot(W_WT1_, np.multiply(vt_,np.reciprocal(np.dot(ht, W_) + 1e-10)))
        #         ht_out = np.multiply(ht, aux) + const
        #         squared_norm = np.multiply(np.dot(ht_out - ht, ht_out - ht), np.reciprocal(np.dot(ht, ht)))
        #         ht[:] = ht_out
        
        for vt, ht in zip(Vt, Ht):
            
            try:
                idx=idx.get()
            except:
                pass
            try:
                W_WT1=W_WT1.get()
            except:
                pass
            try:
                W=W.get()
            except:
                pass
            try:
                ht=ht.get()
            except:
                pass
            try:
                vt_=vt_.get()
            except:
                pass
            
            vt_ = vt.data
            idx = vt.indices
            W_WT1_ = W_WT1[:, idx]
            W_ = W[:, idx]
            

            for n_iter_ in range(max_iter):
                R=np.dot(ht, W_)
                S=np.multiply(vt_,np.reciprocal(R + 1e-10))
                aux = np.dot(W_WT1_, S)
                ht_out = np.multiply(ht, aux) + const
                squared_norm = np.multiply(np.dot(ht_out - ht, ht_out - ht), np.reciprocal(np.dot(ht, ht)))
                squared_norm_mask = squared_norm > squared_epsilon
                ht[squared_norm_mask] = ht_out[squared_norm_mask]
                if not np.any(squared_norm_mask):
                    break
>>>>>>>> cu-cat/DT5:cu_cat/_gap_encoder.py
    return Ht

def _multiplicative_update_h_smallfast(
    Vt: np.array,
    W: np.array,
    Ht: np.array,
    epsilon: float = 1e-2,
    max_iter: int = 10,
    rescale_W: bool = False,
    gamma_shape_prior: float = 1.1,
    gamma_scale_prior: float = 1.0,
):
    """
    Multiplicative update step for the activations H.
    """
    if rescale_W:
        WT1 = 1 + 1 / gamma_scale_prior
        W_WT1 = W / WT1
    else:
        WT1 = np.sum(W, axis=1) + 1 / gamma_scale_prior
        W_WT1 = W / WT1.reshape(-1, 1)
    const = (gamma_shape_prior - 1) / WT1
    squared_epsilon = epsilon #**2

    squared_norm = 1
    Vt=csr_gpu(Vt);Ht=cp.array(Ht);W=cp.array(W);W_WT1=cp.array(W_WT1.T)#;Vt=cp.array(Vt)
    for n_iter_ in range(max_iter):
        if squared_norm <= squared_epsilon:
            break
        C=Vt.multiply( cp.reciprocal(cp.matmul(Ht, W) + 1e-10)) ##sparse now
        aux=C.dot(W_WT1)
        ht_out = cp.multiply(Ht,aux) + const
        # squared_norm = cp.sum(cp.multiply(ht_out - Ht, ht_out - Ht) / cp.multiply(Ht, Ht))
        squared_norm = cp.sum((ht_out - Ht)**2 / (Ht**2))
        Ht = ht_out

    return Ht#.get()

def batch_lookup(
    lookup: NDArray,
    n: int = 1,
) -> Generator[tuple[NDArray, NDArray], None, None]:
    """
    Make batches of the lookup array.
    """
    len_iter = len(lookup)
    for idx in range(0, len_iter, n):
        indices = lookup[slice(idx, min(idx + n, len_iter))]
        unq_indices = np.unique(indices)
        yield unq_indices, indices


def get_kmeans_prototypes(
    X: ArrayLike,
    n_prototypes: int,
    analyzer: Literal["word", "char", "char_wb"] = "char",
    hashing_dim: int = 128,
    ngram_range: tuple[int, int] = (2, 4),
    sparse: bool = False,
    sample_weight=None,
    random_state: int | RandomState | None = None,
) -> NDArray:
    """
    Computes prototypes based on:
      - dimensionality reduction (via hashing n-grams)
      - k-means clustering
      - nearest neighbor
    """
    vectorizer = self._HV(
        analyzer=analyzer,
        norm=None,
        alternate_sign=False,
        ngram_range=ngram_range,
        n_features=hashing_dim,
    )
    projected = vectorizer.transform(X)
    if not sparse:
        projected = projected.toarray()
    kmeans = KMeans(n_clusters=n_prototypes, n_init=10, random_state=random_state)
    kmeans.fit(projected, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    neighbors = NearestNeighbors()
    neighbors.fit(projected)
    indexes_prototypes = np.unique(neighbors.kneighbors(centers, 1)[-1])
    return np.sort(X[indexes_prototypes])
