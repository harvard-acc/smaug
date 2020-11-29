Internal modules
================

This page describes internal APIs that can be used to add new features to
SMAUG's Python API. These are *not* meant to be used for building DL models
using SMAUG.

Building new operators
----------------------

.. currentmodule:: smaug.python.ops.common

.. autofunction:: add_node

.. currentmodule:: smaug.python.ops.array_ops
.. autofunction:: broadcast_inputs
.. autofunction:: check_and_add_layout_transform
