import os
import unittest
import tempfile
import shutil
import subprocess
import numpy as np
from numpy.testing import assert_array_almost_equal

from smaug.python.global_vars import *
from smaug.core.tensor_pb2 import *

class TensorflowTest(unittest.TestCase):
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
      returncode = subprocess.call(cmd, shell=True, stdout=True, stderr=f)

    if returncode != 0:
      print("\nTEST FAILED! Contents of stderr:")
      print("--------------------------------\n")
      with open(self.error_filename, "r") as f:
        for line in f:
          print(line.strip())
      print("--------------------------------")

    return returncode

  def runAndValidate(self, graph, tf_output):
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
    sg_output = np.reshape(sg_output, sg_output_proto.shape.dims)
    assert_array_almost_equal(tf_output, sg_output, decimal=3)
