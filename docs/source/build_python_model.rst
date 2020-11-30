Build a SMAUG model with Python API
===================================

SMAUG's Python frontend provides easy APIs to build DL models. The created model
is a computational graph, which is serialized into two protobuf files, one for
the model topology and the other for the parameters. These two files are inputs
to SMAUG's C++ runtime that performs the actual simulation. In this tutorial,
we will be using the SMAUG Python APIs to build new DL models.

Before building a model, we need to create a `Graph` context in which we will
add operators.

.. code-block:: python

    import smaug as sg
    with sg.Graph(name="my_model", backend="SMV") as graph:
      # Any operators instantiated within the context will be added to `graph`.

When using the :class:`smaug.Graph` API to create a graph context, we need to
give a name for the model and select the backend we want to use when running the
model through the C++ runtime. A backend in SMAUG is a logical combination of
hardware blocks that implements all the SMAUG operators. Refer to
`C++ docs <doxygen_html/index.html>`_ for more details of a backend
implementation in the C++ runtime. Here, we choose the :code:`SMV` backend that
comes with SMAUG, which is modeled after the NVDLA architecture. SMAUG also has
another backend named :code:`Reference`, which is a reference implementation
without having specific performance optimizations. Also, refer to
:class:`smaug.Graph` for a detailed description of the parameters.

Now we can start adding operators to the graph context. The following gives an
example of building a simple 3-layer model.

.. code-block:: python

    import numpy as np
    import smaug as sg

    def generate_random_data(shape):
      r = np.random.RandomState(1234)
      return (r.rand(*shape) * 0.005).astype(np.float16)

    with sg.Graph(name="my_model", backend="SMV") as graph:
      input_tensor = sg.Tensor(
          data_layout=sg.NHWC, tensor_data=generate_random_data((1, 28, 28, 1)))
      conv_weights = sg.Tensor(
          data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 1)))
      fc_weights = sg.Tensor(
          data_layout=sg.NC, tensor_data=generate_random_data((10, 6272)))

      # Shape of act: [1, 28, 28, 1].
      act = sg.input_data(input_tensor)
      # After the convolution, shape of act: [1, 32, 28, 28].
      act = sg.nn.convolution(
          act, conv_weights, stride=[1, 1], padding="same", activation="relu")
      # After the max pooling, shape of act: [1, 32, 14, 14].
      act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
      # After the matrix multiply, shape of act: [1, 10].
      act = sg.nn.mat_mul(act, fc_weights)

As we create the first operator for the model, we need to first prepare an
input tensor and weight tensors that are used by the operator. Tensors are
represented by the :class:`smaug.Tensor`. In the example, we create an input
tensor :code:`input_tensor` using the API. Here, we specify the
:code:`data_layout` as :code:`sg.NHWC`, which stands for a 4D tensor shape with
the channel-major layout. We also specify the :code:`tensor_data` parameter
with a randomly generated NumPy array, with a shape of :code:`[1, 28, 28, 1]`.
However, the user can use the real weights extracted from a pretrained model.
Likewise, we create two weight tensors that will be used by a convolution
operator and a matrix multiply operator, respectively.

A :func:`smaug.data_op`, which simply forwards an input tensor to its output, is
required for any tensor that is not the output of another operator. Here,
:code:`act` is a reference to code:`input_tensor`. Then, :code:`act` is
fed to a convolution operator that also takes :code:`conv_weights` as its
filter input. With more details provided in :func:`smaug.nn.convolution`, it
computes a 3D convolution given the 4D input and filter tensors, and we use 1x1
strides, the :code:`same` padding and a ReLU activation fused with the
convolution operation.  The output of it then goes through a max pooling
operator with a 2x2 filter size, which in turn fans its output into the last
matrix multiply operator. Note that since the output of the max pooling
operator is a 4D tensor while :func:`smaug.nn.mat_mul` expects a 2D input
tensor, SMAUG will automatically add a layout transformation operator
:func:`smaug.tensor.reorder` in between to make the data layout format
compatible. Thus, the 4D tensor of shape :code:`[1, 32, 14, 14]` will be
flattened into a 2D tensor of shape :code:`[1, 6272]` before running the matrix
multiply. Similarly, SMAUG will also perform the NHWC to NCHW layout
transformation or vice versa as per the expected layout format of the backend.

After finishing adding operators to the model, we can now take a look at the
summary of the model using the :func:`smaug.Graph.print_summary` API.

.. code-block:: python

    graph.print_summary()

This prints model-level information and operator-specific properties as below::

  =================================================================
  Summary of the network: my_model (SMV)
  =================================================================
  Host memory access policy: AllDma.
  -----------------------------------------------------------------
  Name: data (Data)
  Parents:
  Children:conv
  Input tensors:
    data/input0 Float16 [1, 28, 28, 1] NHWC alignment(8)
  Output tensors:
    data/output0 Float16 [1, 28, 28, 1] NHWC alignment(8)
  -----------------------------------------------------------------
  Name: data_1 (Data)
  Parents:
  Children:conv
  Input tensors:
    data_1/input0 Float16 [32, 3, 3, 1] NHWC alignment(8)
  Output tensors:
    data_1/output0 Float16 [32, 3, 3, 1] NHWC alignment(8)
  -----------------------------------------------------------------
  Name: conv (Convolution3d)
  Parents:data data_1
  Children:max_pool
  Input tensors:
    data/output0 Float16 [1, 28, 28, 1] NHWC alignment(8)
    data_1/output0 Float16 [32, 3, 3, 1] NHWC alignment(8)
  Output tensors:
    conv/output0 Float16 [1, 28, 28, 32] NHWC alignment(8)
  -----------------------------------------------------------------
  Name: max_pool (MaxPooling)
  Parents:conv
  Children:reorder
  Input tensors:
    conv/output0 Float16 [1, 28, 28, 32] NHWC alignment(8)
  Output tensors:
    max_pool/output0 Float16 [1, 14, 14, 32] NHWC alignment(8)
  -----------------------------------------------------------------
  Name: reorder (Reorder)
  Parents:max_pool
  Children:mat_mul
  Input tensors:
    max_pool/output0 Float16 [1, 14, 14, 32] NHWC alignment(8)
  Output tensors:
    reorder/output0 Float16 [1, 6272] NC alignment(8)
  -----------------------------------------------------------------
  Name: data_2 (Data)
  Parents:
  Children:mat_mul
  Input tensors:
    data_2/input0 Float16 [10, 6272] NC alignment(8)
  Output tensors:
    data_2/output0 Float16 [10, 6272] NC alignment(8)
  -----------------------------------------------------------------
  Name: mat_mul (InnerProduct)
  Parents:reorder data_2
  Children:
  Input tensors:
    reorder/output0 Float16 [1, 6272] NC alignment(8)
    data_2/output0 Float16 [10, 6272] NC alignment(8)
  Output tensors:
    mat_mul/output0 Float16 [1, 10] NC alignment(8)
  -----------------------------------------------------------------

Finally, we can export the model files using the
:func:`smaug.Graph.write_graph` API.

.. code-block:: python

    graph.write_graph()

This gives us two files named :code:`my_model_topo.pbtxt` and
:code:`my_model_params.pb`, where the former stores all the model information
except for the parameters, which are stored in the latter. This separation is
helpful for us to quickly check things in the human readable topology file
while still compressing as much as possible the oftentimes large paramaters.
We can now move on to the `C++ side tutorials <doxygen_html/index.html>`_ that
explain the details of using these two files to run the model.
