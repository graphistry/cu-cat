`cu_cat`

.. .. image:: https://cu_cat-data.github.io/stable/_static/cu_cat.svg
..    :align: center
..    :width: 50 %
..    :alt: cu_cat logo


.. |py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |black|

.. .. |py_ver| image:: https://img.shields.io/pypi/pyversions/cu_cat
.. .. |pypi_var| image:: https://img.shields.io/pypi/v/cu_cat?color=informational
.. .. |pypi_dl| image:: https://img.shields.io/pypi/dm/cu_cat
.. .. |codecov| image:: https://img.shields.io/codecov/c/github/cu_cat-data/cu_cat/main
.. .. |circleci| image:: https://img.shields.io/circleci/build/github/cu_cat-data/cu_cat/main?label=CircleCI
.. .. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg


`cu_cat <https://cu_cat-data.github.io/>`_ (a fork of skrub enhanced with CUDA gpu magics, formerly *dirty_cat*) is a Python
library that facilitates prepping your tables for machine learning.

If you like the package, spread the word and ⭐ this repository!

What can `cu_cat` do?
--------------------

`cu_cat` provides data assembling tools (``TableVectorizer``, ``fuzzy_join``...) and
encoders (``GapEncoder``, ``MinHashEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

See our `examples <https://cu_cat-data.org/stable/auto_examples>`_.

What `cu_cat` cannot do
~~~~~~~~~~~~~~~~~~~~~~
=======
===========

`cu_cat` is an end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics.
It is a CUDA accelerated port of what was dirty_cat, now rebranded as `skrub <https://github.com/skrub-data/skrub>`_

What can `cu_cat` do?
------------------------

`cu_cat` provides tools (``TableVectorizer``...) and
encoders (``GapEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

`Example notebooks <https://github.com/graphistry/cu-cat/tree/master/examples/cu-cat_demo.ipynb>`_
goes in-depth on how to identify and deal with dirty data (biological in this case) using the `cu_cat` library.

What `cu_cat` does not
~~~~~~~~~~~~~~~~~~~~~~~~~

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

`cu_cat` can still help with handling typos and variations in this kind of setting.

For a detailed description of the problem of encoding dirty categorical data, see
`Similarity encoding for learning with dirty categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_
and `Encoding high-cardinality string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

cu_cat v 0.04 can be easily installed via `pip`::

    pip install git+http://github.com/graphistry/cu-cat.git@v0.04.0

Dependencies
~~~~~~~~~~~~

Major dependencies the cuml and cudf libraries, as well as `standard python libraries <https://github.com/skrub-data/skrub/blob/main/setup.cfg>`_

Related projects
----------------

dirty_cat is now rebranded as part of the sklearn family as `skrub <https://github.com/skrub-data/skrub>`_

