import numpy as np
from tensor_pb2 import *
from types_pb2 import *
from global_vars import *

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

    # Do data padding if this Tensor contains data.
    if self.tensor_data is not None:
      pad_width = [(0, 0) for i in xrange(len(dims) - 1)]
      pad_width.append((0, self.calc_padding(dims[-1])))
      tensor_data = np.pad(tensor_data, pad_width, 'constant')

  # This returns the size we need to pad on the last dimension.
  def calc_padding(self, value):
    if self.shape.alignment == 0 or value % self.shape.alignment == 0:
      return 0
    return (self.shape.alignment - (value % self.shape.alignment))

  def to_tensor_proto(self, tensor_proto):
    tensor_proto.name = self.name
    tensor_proto.shape.CopyFrom(self.shape)
    tensor_proto.data_type = self.data_type
    tensor_proto.data_format = self.data_format
    if self.tensor_data is not None:
      for x in np.nditer(self.tensor_data):
        if self.data_type == Float16:
          tensor_proto.half_data.append(x)
        elif self.data_type == Float32:
          tensor_proto.float_data.append(x)
        elif self.data_type == Float64:
          tensor_proto.double_data.append(x)
        elif self.data_type == Int32:
          tensor_proto.int_data.append(x)
        elif self.data_type == Int64:
          tensor_proto.int64_data.append(x)
