import numpy as np

from smaug.core import types_pb2

np_to_smaug_type = {
    np.float16: types_pb2.Float16,
    np.float32: types_pb2.Float32,
    np.float64: types_pb2.Float64,
    np.int32: types_pb2.Int32,
    np.int64: types_pb2.Int64,
    np.bool_: types_pb2.Bool,
}

class LayoutSet:
  def __init__(self, bitmask=0):
    self.layouts = bitmask

  def __eq__(self, other):
    return self.layouts == other.layouts

  def __lt__(self, other):
    return self.layouts < other.layouts

  def insert(self, layout):
    self.layouts |= layout

  def remove(self, layout):
    self.layouts &= (~layout)

  def contains(self, layout):
    return (self.layouts >= layout and self.layouts & layout != 0)

  def overlaps_with(self, other):
    return (self.layouts & other.layouts) != 0

class OperatorLayouts:
  def __init__(self, input_bitmasks, output_bitmask):
    self.input_layoutsets = [LayoutSet(bitmask) for bitmask in input_bitmasks]
    self.output_layoutset = LayoutSet(output_bitmask)
