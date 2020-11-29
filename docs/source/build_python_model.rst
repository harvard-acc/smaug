Build a SMAUG model with Python API
===================================

SMAUG's Python frontend provides easy APIs to build DL models. The created model
is a computational graph, which are serialized into two protobuf files, one for
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
implmentation in the C++ runtime. Here, we choose the `SMV` backend that comes
with SMAUG, which is modeled after the NVDLA architecture. Also, refer to
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
      conv_tensor = sg.Tensor(
          data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 1)))
      fc_tensor = sg.Tensor(
          data_layout=sg.NC, tensor_data=generate_random_data((10, 6272)))

      # Shape of act: [1, 28, 28, 1].
      act = sg.input_data(input_tensor)
      # After the convolution, shape of act: [1, 32, 28, 28].
      act = sg.nn.convolution(
          act, conv_tensor, stride=[1, 1], padding="same", activation="relu")
      # After the max pooling, shape of act: [1, 32, 14, 14].
      act = sg.nn.max_pool(act, pool_size=[2, 2], stride=[2, 2])
      # After the matrix multiply, shape of act: [1, 10].
      act = sg.nn.mat_mul(act, fc_tensor)

As we create the first operator for the model, we need to first prepare an input
tensor and weight tensors that are used by the operator. In the example, we
create an input tensor `input_tensor` using the :class:`smaug.Tensor` API. Here,
we specify the `data_layout` as `sg.NHWC`, which stands for a 4D tensor shape
with the channel-major layout. We also specify the `tensor_data` parameter with
a randomly generated NumPy array, with a shape of [1, 28, 28, 1]. However, the
user can use the real weights extracted from a pretrained model. Likewise, we
create two weight tensors that will be used by a convolution operator and a
matrix multiply operator, respectively.

The `sg.input_data` creates a data operator that simply creates a reference
tensor `act` to its input data `input_tensor`. Then, `act` is fed to a
convolution operator that also takes `conv_tensor` as its filter input. With
more details provided in :func:`smaug.nn.convolution`, it computes a 3D
convolution given the 4D input and filter tensors, and we use 1x1 strides,
the `same` padding and a ReLU activation fused with the convolution operation.
The output of it then goes through a max pooling operator with a 2x2 filter
size, which in turn fans its output into the last matrix multiply operator. Note
that since the output of the max pooling operator is a 4D tensor while
:func:`smaug.nn.mat_mul` expects a 2D input tensor, SMAUG will automatically
add a layout transformation operator :func:`smaug.tensor.reorder` in between to
make the data layout format compatible. Thus, the 4D tensor of shape
[1, 32, 14, 14] will be flattened into a 2D tensor of shape [1, 6272] before
running the matrix multiply. Similarly, SMAUG will also perform the NHWC to NCHW
layout transformation or vice versa as per the expected layout format of the
backend.

After finishing adding operators to the model, we can now take a look at the
summary of the model using the :func:`smaug.Graph.print_summary` API, which
prints model-level information and operator-specific properties.

.. code-block:: python

    graph.print_summary()

Finally, we can export the model files using the
:func:`smaug.Graph.write_graph` API.

.. code-block:: python

    graph.write_graph()

This gives us two files named `my_model_topo.pbtxt` and `my_model_params.pb`,
where the former stores all the model information except for the parameters,
which are stored in the latter. This separation is helpful for us to quickly
check things in the human readable topology file while still compressing as
much as possible the oftentimes large paramaters.
