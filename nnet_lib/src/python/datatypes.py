import numpy as np
from types_pb2 import *

np_to_smaug_type = {
    np.float16: Float16,
    np.float32: Float32,
    np.float64: Float64,
    np.int32: Int32,
    np.int64: Int64
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
  def __init__(self, input_bitmask, output_bitmask):
    self.input_layoutset = LayoutSet(input_bitmask)
    self.output_layoutset = LayoutSet(output_bitmask)
