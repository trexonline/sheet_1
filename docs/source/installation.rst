Installation
============

To install this package you need to clone the repository using

.. code-block:: bash

    git clone git@github.com:trexonline/sheet_1.git

To set up a virtual environment inside the repository install uv

.. code-block:: bash

    pip install uv

and create it with

.. code-block:: bash

    uv sync

This package uses numpy and qiskit which can be installed via the following command lines:

.. code-block:: bash

    uv add numpy
    uv add qiskit

