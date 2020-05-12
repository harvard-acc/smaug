import numpy as np

from smaug.core.tensor_pb2 import *
from smaug.core.types_pb2 import *
from smaug.python.global_vars import *
from smaug.python.datatypes import *

class Tensor:
  def __init__(self,
               dims=None,
               name=None,
               data_layout=NCHW,
               data_type=None,
               data_format=Uncompressed,
               tensor_data=None,
               source=None,
               alignment=None):
    self.shape = TensorShapeProto()
    self.tensor_data = tensor_data
    # If tensor_data is provided, deduce dims and data_type directly from it
    # (the kwargs are ignored if they are provided).
    if self.tensor_data is not None:
      self.deduce_attrs_from_data()
    else:
      self.shape.dims.extend(dims)
      self.data_type = data_type

    self.shape.layout = data_layout
    self.name = name
    self.data_format = data_format
    self.source = source
    if alignment != None:
      self.shape.alignment = alignment
    elif get_graph() == None:
      self.shape.alignment = 0
    else:
      self.shape.alignment = get_graph().alignment

    # Do data padding if this Tensor contains data.
    if self.tensor_data is not None:
      pad_width = [(0, 0) for i in range(len(self.shape.dims) - 1)]
      pad_width.append((0, self.calc_padding(self.shape.dims[-1])))
      self.tensor_data = np.pad(self.tensor_data, pad_width, 'constant')

  def dims(self, index):
    """This returns the size of the dimension."""
    assert index < len(self.shape.dims), "The dimension index is out of bound!"
    return self.shape.dims[index]

  def deduce_attrs_from_data(self):
    """Deduce tensor attributes from the supplied tensor data.

    The deducible attributes include tensor shape dimensions and data type.
    """
    # Deduce dims from tensor data.
    self.shape.dims.extend(list(self.tensor_data.shape))
    # Deduce data type from tensor data
    try:
      self.data_type = np_to_smaug_type[self.tensor_data.dtype.type]
    except KeyError:
      assert False, "We don't support numpy dtype: %s" % self.tensor_data.dtype

  def calc_padding(self, value):
    """This returns the size we need to pad on the last dimension."""
    if self.shape.alignment == 0 or value % self.shape.alignment == 0:
      return 0
    return (self.shape.alignment - (value % self.shape.alignment))

  def to_tensor_proto(self, tensor_proto, tensor_data_array):
    """Serialize the tensor into a tensor proto.

    Args:
      tensor_proto: The tensor proto this tensor gets serialized into.
      tensor_data_array: The tensor data array this tensor gets serialized into.
    """
    tensor_proto.name = self.name
    tensor_proto.shape.CopyFrom(self.shape)
    tensor_proto.data_type = self.data_type
    tensor_proto.data_format = self.data_format
    if self.tensor_data is not None:

      # Since Protobuf doesn't support float16 data type, we pack two float16
      # elements into one int32.
      if self.data_type == Float16:
        # Numpy.view comes in handy here. Note that it won't work if
        # tensor_data's last dimension is of odd size. To solve that, we
        # flatten the tensor data, and if the flattened list is still of
        # odd size, we pad a zero at the end of the list. When we later
        # deserialize the tensor data, we know the correct shape of the
        # tensor, and the padded zero will be discarded.
        self.tensor_data = self.tensor_data.flatten()
        if self.tensor_data.size % 2 != 0:
          self.tensor_data = np.append(self.tensor_data, np.float16(0))
        self.tensor_data = self.tensor_data.view(np.int32)

      # Serialize the data into the proto.
      tensor_data_proto = tensor_data_array.data_array.add()
      tensor_data_proto.name = tensor_proto.name
      data_list = [x for x in np.nditer(self.tensor_data)]
      if self.data_type == Float16:
        tensor_data_proto.half_data.extend(data_list)
      elif self.data_type == Float32:
        tensor_data_proto.float_data.extend(data_list)
      elif self.data_type == Float64:
        tensor_data_proto.double_data.extend(data_list)
      elif self.data_type == Int32:
        tensor_data_proto.int_data.extend(data_list)
      elif self.data_type == Int64:
        tensor_data_proto.int64_data.extend(data_list)
