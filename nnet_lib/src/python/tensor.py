import numpy as np
from tensor_pb2 import *
from types_pb2 import *
from global_vars import *
from datatypes import *

class Tensor:
  def __init__(self,
               dims,
               name=None,
               data_layout=NCHW,
               data_type=Float32,
               data_format=Uncompressed,
               tensor_data=None,
               source=None,
               target=None,
               alignment=None):
    self.shape = TensorShapeProto()
    self.shape.dims.extend(dims)
    self.shape.layout = data_layout
    self.name = name
    self.data_type = data_type
    self.data_format = data_format
    self.tensor_data = tensor_data
    self.source = source
    self.target = target
    if alignment != None:
      self.shape.alignment = alignment
    elif get_graph() == None:
      self.shape.alignment = 0
    else:
      self.shape.alignment = get_graph().alignment

    self.check_data_type()
    # Do data padding if this Tensor contains data.
    if self.tensor_data is not None:
      pad_width = [(0, 0) for i in xrange(len(dims) - 1)]
      pad_width.append((0, self.calc_padding(dims[-1])))
      self.tensor_data = np.pad(self.tensor_data, pad_width, 'constant')

  def check_data_type(self):
    """Sanity check on the data type of the tensor data."""
    if self.tensor_data is None:
      return
    try:
      expected_type = smaug_to_np_type[self.data_type]
      assert self.tensor_data.dtype == expected_type
    except KeyError:
      assert False, "Unknown data type!"

  def calc_padding(self, value):
    """This returns the size we need to pad on the last dimension."""
    if self.shape.alignment == 0 or value % self.shape.alignment == 0:
      return 0
    return (self.shape.alignment - (value % self.shape.alignment))

  def to_tensor_proto(self, tensor_proto):
    """Serialize the tensor into a tensor proto.

    Args:
      tensor_proto: The tensor proto this tensor gets serialized into.
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
      data_list = [x for x in np.nditer(self.tensor_data)]
      if self.data_type == Float16:
        tensor_proto.half_data.extend(data_list)
      elif self.data_type == Float32:
        tensor_proto.float_data.extend(data_list)
      elif self.data_type == Float64:
        tensor_proto.double_data.extend(data_list)
      elif self.data_type == Int32:
        tensor_proto.int_data.extend(data_list)
      elif self.data_type == Int64:
        tensor_proto.int64_data.extend(data_list)
