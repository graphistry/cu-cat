<<<<<<< HEAD
`skrub`
=======

.. image:: https://skrub-data.github.io/stable/_static/skrub.svg
   :align: center
   :width: 50 %
   :alt: skrub logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |black|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/skrub
.. |pypi_var| image:: https://img.shields.io/pypi/v/skrub?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/skrub
.. |codecov| image:: https://img.shields.io/codecov/c/github/skrub-data/skrub/main
.. |circleci| image:: https://img.shields.io/circleci/build/github/skrub-data/skrub/main?label=CircleCI
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg


`skrub <https://skrub-data.github.io/>`_ (formerly *dirty_cat*) is a Python
library that facilitates prepping your tables for machine learning.

If you like the package, spread the word and ⭐ this repository!

What can `skrub` do?
--------------------

`skrub` provides data assembling tools (``TableVectorizer``, ``fuzzy_join``...) and
encoders (``GapEncoder``, ``MinHashEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

See our `examples <https://skrub-data.org/stable/auto_examples>`_.

What `skrub` cannot do
~~~~~~~~~~~~~~~~~~~~~~
=======
`cu_cat`
===========

`cu_cat` is an end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics.
It is a cuda accelerated port of what was dirty_cat, now rebranded as `skrub <https://github.com/skrub-data/skrub>`_

What can `cu_cat` do?
------------------------

`cu_cat` provides tools (``TableVectorizer``...) and
encoders (``GapEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

`Example notebooks <https://github.com/graphistry/cu-cat/tree/master/examples/cu-cat_demo.ipynb>`_
goes in-depth on how to identify and deal with dirty data (biological in this case) using the `cu_cat` library.

What `cu_cat` does not
~~~~~~~~~~~~~~~~~~~~~~~~~
>>>>>>> cu-cat/DT5

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

<<<<<<< HEAD
`skrub` can still help with handling typos and variations in this kind of setting.
=======
`cu_cat` can still help with handling typos and variations in this kind of setting.
>>>>>>> cu-cat/DT5

For a detailed description of the problem of encoding dirty categorical data, see
`Similarity encoding for learning with dirty categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_
and `Encoding high-cardinality string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

<<<<<<< HEAD
Installation (WIP)
------------------

There are currently no PiPy releases.
You can still install the package from the GitHub repository with:

.. code-block:: shell

    pip install git+https://github.com/skrub-data/skrub.git

=======
cu_cat v 0.04 can be easily installed via `pip`::

    pip install git+http://github.com/graphistry/cu-cat.git@v0.04.0
>>>>>>> cu-cat/DT5

Dependencies
~~~~~~~~~~~~

<<<<<<< HEAD
Dependencies and minimal versions are listed in the `setup <https://github.com/skrub-data/skrub/blob/main/setup.cfg#L27>`_ file.
=======
Major dependencies the cuml and cudf libraries, as well as `standard python libraries <https://github.com/skrub-data/skrub/blob/main/setup.cfg>`_
>>>>>>> cu-cat/DT5

Related projects
----------------

<<<<<<< HEAD
Are listed on the `skrub's website <https://skrub-data.github.io/stable/#related-projects>`_

Contributing
------------

The best way to support the development of skrub is to spread the word!

Also, if you already are a skrub user, we would love to hear about your use cases and challenges in the `Discussions <https://github.com/skrub-data/skrub/discussions>`_ section.

To report a bug or suggest enhancements, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ and/or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.

Additional resources
--------------------

* `Introductory video (YouTube) <https://youtu.be/_GNaaeEI2tg>`_
* `Overview poster for EuroSciPy 2023 (Dropbox) <https://www.dropbox.com/scl/fi/89tapbshxtw0kh5uzx8dc/Poster-Euroscipy-2023.pdf?rlkey=u4ycpiyftk7rzttrjll9qlrkx&dl=0>`_

References
----------

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
=======
dirty_cat is now rebranded as part of the sklearn family as `skrub <https://github.com/skrub-data/skrub>`_

>>>>>>> cu-cat/DT5
