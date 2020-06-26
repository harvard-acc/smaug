"""Implements test fixtures and common functions for SMAUG tests."""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
import numpy as np
from numpy.testing import assert_array_almost_equal

from smaug.python.global_vars import *
from smaug.core.tensor_pb2 import *

def get_tensor_data(tensor_data_array, tensor_name):
  """ Find the tensor data for this tensor by its name."""
  for i in range(len(tensor_data_array.data_array)):
    if tensor_data_array.data_array[i].name == tensor_name:
      return tensor_data_array.data_array[i]
  return None

def _account_for_padding(shape):
  alignment = shape.alignment
  remainder = shape.dims[-1] % shape.alignment
  if alignment == 0 or remainder == 0:
    return shape
  shape.dims[-1] += alignment - remainder;
  return shape

class SmaugTest(unittest.TestCase):
  def setUp(self):
    self.run_dir = tempfile.mkdtemp()
    self.error_filename = os.path.join(self.run_dir, "stderr")
    self.graph_name = "test_graph"
    self.backend = "SMV"
    self.binary = os.environ["SMAUG_HOME"] + "/build/bin/smaug"
    self.dtype = backend_datatype[self.backend]

  def tearDown(self):
    """ Delete temporary files and outputs. """
    shutil.rmtree(self.run_dir)

  def launchSubprocess(self, cmd):
    with open(self.error_filename, "w") as f:
      returncode = subprocess.call(cmd, shell=True, stdout=None, stderr=f)

    if returncode != 0:
      print("\nTEST FAILED! Contents of stderr:")
      print("--------------------------------\n")
      with open(self.error_filename, "r") as f:
        for line in f:
          print(line.strip())
      print("--------------------------------")

    return returncode

  def runAndValidate(self, graph, expected_output):
    """ Run the test and validate the results. """
    os.chdir(self.run_dir)
    graph.write_graph()
    cmd = "%s %s_topo.pbtxt %s_params.pb --print-last-output=proto" % (
        self.binary, self.graph_name, self.graph_name)
    returncode = self.launchSubprocess(cmd)
    self.assertEqual(returncode, 0, msg="Test returned nonzero exit code!")

    # Read the SMAUG result and validate it against the TF result.
    sg_output_proto = TensorProto()
    with open("output.pb", "rb") as f:
      sg_output_proto.ParseFromString(f.read())
    sg_output = None
    if backend_datatype[self.backend] == np.float16:
      sg_output = np.array(
          sg_output_proto.data.half_data, dtype=np.int32).view(np.float16)
    elif backend_datatype[self.backend] == np.float32:
      sg_output = sg_output_proto.data.float_data
    elif backend_datatype[self.backend] == np.float64:
      sg_output = sg_output_proto.data.double_data
    elif backend_datatype[self.backend] == np.int32:
      sg_output = sg_output_proto.data.int_data
    elif backend_datatype[self.backend] == np.int64:
      sg_output = sg_output_proto.data.int64_data
    shape = _account_for_padding(sg_output_proto.shape)
    sg_output = np.reshape(sg_output, sg_output_proto.shape.dims)
    assert_array_almost_equal(expected_output, sg_output, decimal=3)
