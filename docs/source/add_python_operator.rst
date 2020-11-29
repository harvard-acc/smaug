Adding a new operator
=====================

.. module:: smaug.python.ops

To add a new operator to SMAUG, we first need to implement the actual operator
in C++.  Please see the `C++ tutorial <doxygen_html/index.html>`_ for details.
Once that is done, all that is left is to expose a Python API which will add
the operator to the model.

Because at their core, Python operators simply add nodes to the graph, we can
build arbitrarily complex operators in Python that invoke other Python
operators. We will start simple, and then demonstrate more complex examples.

Adding a simple Python operator
-------------------------------

A simple Python operator is one that adds only a single underlying SMAUG
operator. For the purposes of this tutorial, we'll continue using the
:code:`MyCustomOperator` operator we built in the C++ tutorial, which
implements an elementwise add. This operator is as simple as it can get: it
takes just two input tensors and no additional parameters and produces a single
output tensor.

First, go to :code:`smaug/python/ops`. All operators are defined in Python
files here, organized by operator type. For example, :code:`math_ops` contains
arithmetic operators like add/multiply/etc, while :code:`array_ops` contains
operators for manipulating tensors by reshaping/concatentating/etc.  In
practice, we would add a new operator to an existing file there. But for the
purposes of this tutorial, we'll create a brand new file called
`my_custom_operator.py`.

A SMAUG Python operator takes some number of input tensors, additional operator
parameters (if appropriate), and returns some number of output tensors. Input
and output tensors are all represented as :class:`smaug.Tensor` objects. The
work that's done is to add a properly formed :code:`NodeProto` to the
:code:`GraphProto` with the :func:`smaug.python.ops.common.add_node` API.
Here's how the `my_custom_operator.py` file might look:

.. code-block:: python

   from smaug.core import node_pb2, types_pb2
   from smaug.python.ops import common

   def my_custom_operator(tensor_a, tensor_b, name="my_custom_operator"):
     if tensor_a.shape.dims != tensor_b.shape.dims:
       raise ValueError(
           "The input tensors to MyCustomOperator must be of the same shape")
     return common.add_node(
       name=name,
       op=types_pb2.MyCustomOperator,
       input_tensors=[tensor_a, tensor_b],
       output_tensors_dims=[tensor_a.shape.dims],
       output_tensor_layout=tensor_a.shape.layout)[0]

The :code:`name` parameter is actually a prefix for the actual name of the node
that's added to the graph. SMAUG will automatically generate a unique suffix to
ensure that no two nodes have the same name. Also, :func:`common.add_node`
returns a list of tensors, but in our case, we have only one, so to simplify
our API, we just return the first element.  There are other optional parameters
to :func:`common.add_node`, but they aren't needed in this basic scenario.

The final step is to expose this operator at the global :py:mod:`smaug` module
level. Open up :file:`smaug/__init__.py` and add the following line:

.. code-block:: python

   from smaug.python.ops import my_custom_operator

And that's it! You're now ready to use this new operator in a new model. Users
will refer to it as :code:`smaug.my_custom_operator.my_custom_operator`.
Obviously, the name for this small example is quite repetitive and
uninformative, but in practice, you would use a more descriptive module
and operator name.

Adding an operator with additional parameters
---------------------------------------------

Some operators require additional parameters beyond just the input tensors. For
example, you may want to specify padding or stride lengths to a convolution. If
your operator needs additional parameters, you will need to add a custom
parameter protobuf message to store them. Open `smaug/core/node.proto
<doxygen_html/node_8proto_source.html>`_.  This file contains the
:code:`NodeProto` definition along with various operator-specific parameters.
Then follow these steps.

1. Define a new message to store your operator's parameters.
2. Add it as a :code:`oneof` field in the :code:`Params` message.
3. Build a :code:`Params` proto in your Python operator, populate it, and pass
   it to :func:`common.add_node`.

As an example, suppose our custom operator actually performed the operation A +
x*B, where x is a user-defined scalar. Then we would add a parameter message
like so:

.. code-block:: c
   :emphasize-lines: 1-3,8

   message MyCustomOperatorParams {
     float scale_factor = 1;
   }

   message Params {
     oneof value {
       # ... if we already have five other parameters already here...
       MyCustomOperatorParams my_custom_operator_params = 6;
     }
     # ... anything else already here ...
   }

.. code-block:: python

   def my_custom_operator(tensor_a, tensor_b, scale_factor=1.0 name=None):
     if tensor_a.shape.dims != tensor_b.shape.dims:
       raise ValueError(
           "The input tensors to MyCustomOperator must be of the same shape")
     params = node_pb2.Params()
     params.my_custom_operator_params.scale_factor = scale_factor
     return common.add_node(
       name=name,
       op=types_pb2.MyCustomOperator,
       input_tensors=[tensor_a, tensor_b],
       output_tensors_dims=[tensor_a.shape.dims],
       output_tensor_layout=tensor_a.shape.layout,
       params=params)[0]

Adding a complex Python operator
--------------------------------

Since Python operators simply add nodes to the graph, we can call Python
operators from each other. As a very simple example, we can chain together
two instances of MyCustomOperator:

.. code-block:: python

   def my_custom_operator_chained(
       tensor_a, tensor_b, scale_factor=1.0 name="my_custom_operator_chained"):
     if tensor_a.shape.dims != tensor_b.shape.dims:
       raise ValueError(
           "The input tensors to MyCustomOperator must be of the same shape")
     params = node_pb2.Params()
     params.my_custom_operator_params.scale_factor = scale_factor
     output_tensor_1 = my_custom_operator(
         tensor_a, tensor_b, scale_factor, name=name)
     return my_custom_operator(
         output_tensor_1, tensor_b, scale_factor, name=name)
