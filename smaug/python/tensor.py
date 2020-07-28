import numpy as np

from smaug.core import tensor_pb2
from smaug.core import types_pb2
from smaug.python import global_vars
from smaug.python import datatypes

class Tensor:
  def __init__(
      self, dims=None, name=None, data_layout=types_pb2.NCHW, data_type=None,
      data_format=types_pb2.Uncompressed, tensor_data=None, source=None,
      source_index=None, targets=None, alignment=None):
    """Create a tensor.

    Args:
      dims: Dimensions of the tensor shape.
      name: Optional Name of the tensor.
      data_layout: Data layout of the tensor.
      data_type: Data type of the tensor.
      data_format: Data format of the tensor.
      tensor_data: A NumPy array that represents the tensor data.
      source: A `Node` that represents this tensor's source node.
      source_index: An int that represents this tensor's output index in its
        source node.
      targets: A list of nodes that use this tensor as inputs.
      alignment: Data alignment used in the tensor data.

    Returns:
      A `Tensor` object.
    """
    self._shape = tensor_pb2.TensorShapeProto()
    self._tensor_data = tensor_data
    # If tensor_data is provided, deduce dims and data_type directly from it
    # (the kwargs are ignored if they are provided).
    if self._tensor_data is not None:
      self._deduce_attrs_from_data()
    else:
      self._shape.dims.extend(dims)
      self._data_type = data_type

    self._shape.layout = data_layout
    self._name = name
    self._data_format = data_format
    self._source = source
    self._source_index = source_index
    if self._source is not None and self._source_index is None:
      raise ValueError(
          "Please provide this tensor's output index in the source node!")
    self._targets = []
    if alignment != None:
      self._shape.alignment = alignment
    elif global_vars.get_graph() == None:
      self._shape.alignment = 0
    else:
      self._shape.alignment = global_vars.get_graph().alignment

    # Do data padding if this Tensor contains data.
    if self._tensor_data is not None:
      pad_width = [(0, 0) for i in range(len(self._shape.dims) - 1)]
      pad_width.append((0, self.calc_padding(self._shape.dims[-1])))
      self._tensor_data = np.pad(self._tensor_data, pad_width, 'constant')

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, name):
    self._name = name

  @property
  def shape(self):
    return self._shape

  @property
  def data_type(self):
    return self._data_type

  @property
  def data_format(self):
    return self._data_format

  @property
  def tensor_data(self):
    return self._tensor_data

  @property
  def source(self):
    return self._source

  @property
  def source_index(self):
    return self._source_index

  @property
  def targets(self):
    return self._targets

  def dims(self, index):
    """This returns the size of the dimension."""
    assert index < len(self._shape.dims), "The dimension index is out of bound!"
    return self._shape.dims[index]

  def _deduce_attrs_from_data(self):
    """Deduce tensor attributes from the supplied tensor data.

    The deducible attributes include tensor shape dimensions and data type.
    """
    # Deduce dims from tensor data.
    self._shape.dims.extend(list(self._tensor_data.shape))
    # Deduce data type from tensor data
    try:
      self._data_type = datatypes.np_to_smaug_type[self._tensor_data.dtype.type]
    except KeyError:
      assert False, "We don't support numpy dtype: %s" % self._tensor_data.dtype

  def calc_padding(self, value):
    """This returns the size we need to pad on the last dimension."""
    if self._shape.alignment == 0 or value % self._shape.alignment == 0:
      return 0
    return (self._shape.alignment - (value % self._shape.alignment))

  def to_tensor_proto(self, tensor_proto, tensor_data_array=None):
    """Serialize the tensor into a tensor proto.

    Args:
      tensor_proto: The tensor proto this tensor gets serialized into.
      tensor_data_array: The tensor data array this tensor gets serialized into.
    """
    tensor_proto.name = self._name
    tensor_proto.shape.CopyFrom(self._shape)
    tensor_proto.data_type = self._data_type
    tensor_proto.data_format = self._data_format
    if self._tensor_data is not None and tensor_data_array is not None:

      # Since Protobuf doesn't support float16 data type, we pack two float16
      # elements into one int32.
      if self._data_type == types_pb2.Float16:
        # Numpy.view comes in handy here. Note that it won't work if
        # tensor_data's last dimension is of odd size. To solve that, we
        # flatten the tensor data, and if the flattened list is still of
        # odd size, we pad a zero at the end of the list. When we later
        # deserialize the tensor data, we know the correct shape of the
        # tensor, and the padded zero will be discarded.
        self._tensor_data = self._tensor_data.flatten()
        if self._tensor_data.size % 2 != 0:
          self._tensor_data = np.append(self._tensor_data, np.float16(0))
        self._tensor_data = self._tensor_data.view(np.int32)

      # Serialize the data into the proto.
      tensor_data_proto = tensor_data_array.data_array.add()
      tensor_data_proto.name = tensor_proto.name
      data_list = [x for x in np.nditer(self._tensor_data)]
      if self._data_type == types_pb2.Float16:
        tensor_data_proto.half_data.extend(data_list)
      elif self._data_type == types_pb2.Float32:
        tensor_data_proto.float_data.extend(data_list)
      elif self._data_type == types_pb2.Float64:
        tensor_data_proto.double_data.extend(data_list)
      elif self._data_type == types_pb2.Int32:
        tensor_data_proto.int_data.extend(data_list)
      elif self._data_type == types_pb2.Int64:
        tensor_data_proto.int64_data.extend(data_list)
      elif self._data_type == types_pb2.Bool:
        tensor_data_proto.bool_data.extend(data_list)
