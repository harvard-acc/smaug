"""Top level SMAUG module.

All operators and core data structures should be imported either directly here
or under a submodule.
"""

from smaug.core.types_pb2 import *

from smaug.python.ops import math_ops as math
from smaug.python.ops import array_ops as tensor
from smaug.python.ops import nn
from smaug.python.ops.control_flow_ops import merge, switch, cond
from smaug.python.ops.data_op import input_data

from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.node import Node

def _autodoc_for_module():
  """Ensures that the top-level SMAUG module gets an autodoc link."""
  pass
