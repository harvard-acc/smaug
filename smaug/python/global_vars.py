"""Here we define global variables.

Currently it contains:
  1) A global active graph
  2) Alignment information for various backends.
  3) Input/output layouts for operators of various backends.
  4) Supported data types of backends.
"""

import numpy as np

from smaug.core.types_pb2 import *
from smaug.python.datatypes import LayoutSet,OperatorLayouts

# This keeps track of the current active graph. Currently we only support
# one active graph, we can change that if later we need multiple graphs.
active_graph = None

def get_graph():
  """Obtain the current active graph."""
  return active_graph

def set_graph(graph):
  """Set the active graph."""
  global active_graph
  active_graph = graph

def clear_graph():
  """Clear the active graph.

  This will be used when the graph context is cleaned up.
  """
  global active_graph
  active_graph = None

# Alignment information for various backends.
backend_alignment = {"Reference": 0, "SMV": 8}

# Input/output layouts for various backends.
backend_layouts = {
    "Reference": {
        Convolution3d: OperatorLayouts([NCHW, NCHW], NCHW),
        ConvolutionDepthwise: OperatorLayouts([NCHW, NCHW], NCHW),
        MaxPooling: OperatorLayouts([NCHW, NCHW], NCHW),
        AveragePooling: OperatorLayouts([NCHW, NCHW], NCHW),
        InnerProduct: OperatorLayouts([NC, CN], NC),
        BatchNorm: OperatorLayouts([NCHW, NC, NC, NC, NC], NCHW),
        Data: OperatorLayouts([X], X),
        ReLU: OperatorLayouts([X], X),
        LReLU: OperatorLayouts([X], X),
        ELU: OperatorLayouts([X], X),
        SELU: OperatorLayouts([X], X),
        Tanh: OperatorLayouts([X], X),
        HardTanh: OperatorLayouts([X], X),
        Sigmoid: OperatorLayouts([X], X),
        Softmax: OperatorLayouts([NC], NC),
        EltwiseAdd: OperatorLayouts([X], X),
        EltwiseMul: OperatorLayouts([X], X),
    },
    "SMV": {
        Convolution3d: OperatorLayouts([NHWC, NHWC], NHWC),
        ConvolutionDepthwise: OperatorLayouts([NHWC, NHWC], NHWC),
        MaxPooling: OperatorLayouts([NHWC, NHWC], NHWC),
        AveragePooling: OperatorLayouts([NHWC, NHWC], NHWC),
        InnerProduct: OperatorLayouts([NC, NC], NC),
        BatchNorm: OperatorLayouts([NHWC, NC, NC, NC, NC], NHWC),
        Data: OperatorLayouts([X], X),
        ReLU: OperatorLayouts([X], X),
        LReLU: OperatorLayouts([X], X),
        ELU: OperatorLayouts([X], X),
        SELU: OperatorLayouts([X], X),
        Tanh: OperatorLayouts([X], X),
        HardTanh: OperatorLayouts([X], X),
        Sigmoid: OperatorLayouts([X], X),
        Softmax: OperatorLayouts([NC], NC),
        EltwiseAdd: OperatorLayouts([X], X),
        EltwiseMul: OperatorLayouts([X], X),
    }
}

backend_datatype = {"SMV": np.float16, "Reference": np.float32}
