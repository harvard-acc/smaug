#!/usr/bin/env python
#
# Integration test runner for nnet_lib.
#
# Usage:
#
#   python run_tests.py path/to/executable

import argparse
import sys
import os
import subprocess
import unittest
import tempfile
import shutil
import re

# Floating point equality comparison tolerance in percent.
FP_ERR = 0.05

# These get set by the command line argument.
BINARY = ""
ARCH = ""

MODEL_DIR = "../../models/"
CORRECT_OUTPUT_DIR = "correct"
OUTPUT_LABELS = "output_labels.out"

class BaseTest(unittest.TestCase):
  def setUp(self):
    self.run_dir = tempfile.mkdtemp()
    self.output_filename = os.path.join(self.run_dir, "stdout")

  def tearDown(self):
    shutil.rmtree(self.run_dir)
    if os.path.exists(OUTPUT_LABELS):
      os.remove(OUTPUT_LABELS)

  def parseOutput(self, output_file):
    test_pred = []
    test_soft = []
    with open(output_file, "r") as f:
      for line in f:
        line = line.strip()
        if "Test" in line:
          # Format is Test N: M, where M is the prediction.
          m = re.findall("\d+", line)
          test_pred.append(int(m[1]))
        else:
          m = re.findall("-?\d+(?:.\d+)", line)
          test_soft.append([float(v) for v in m])
    return test_pred, test_soft

  def launchSubprocess(self, cmd):
    with open(self.output_filename, "w") as f:
      returncode = subprocess.call(
          cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    if returncode != 0:
      print "\nTEST FAILED! Contents of stdout:"
      print "--------------------------------\n"
      with open(self.output_filename, "r") as f:
        for line in f:
          print line.strip()
      print "--------------------------------"

    return returncode

  def almostEqual(self, val, ref):
    """ Returns true if val and ref are within FP_ERR percent of each other. """
    if ((isinstance(val, float) or isinstance(val, int)) and
        (isinstance(ref, float) or isinstance(ref, int))):
      if ref == 0:
        return val == 0
      diff_per = (float(val)-ref)/float(ref) * 100
      if (diff_per < 0):
          diff_per = -diff_per
      return diff_per < FP_ERR
    elif isinstance(val, list) and isinstance(ref, list):
      is_equal = True
      for val_v, ref_v in zip(val, ref):
        is_equal = is_equal and self.almostEqual(val_v, ref_v)
      return is_equal
    else:
      assert("Unsupported types %s, %s for almostEqual comparison" %
             (type(val).__name__, type(ref).__name__))

  def createCommand(self, model_file, correct_output,
                    data_init_mode=None, param_file=None):
    cmd = "%s %s " % (BINARY, os.path.join(MODEL_DIR, model_file))
    if data_init_mode in ["RANDOM", "FIXED", "READ_FILE"]:
      cmd += "-d %s " % data_init_mode
    if param_file and data_init_mode == "READ_FILE":
      cmd += "-f %s " % param_file
    return cmd

  def runAndValidate(self, model_file, correct_output, **kwargs):
    returncode = self.launchSubprocess(
        self.createCommand(model_file, correct_output, **kwargs));

    self.assertEqual(returncode, 0, msg="Test returned nonzero exit code!")

    correct_pred, correct_soft = self.parseOutput(
        os.path.join(CORRECT_OUTPUT_DIR, correct_output))
    test_pred, test_soft = self.parseOutput("output_labels.out")
    for i, (this, correct) in enumerate(zip(test_pred, correct_pred)):
      if this != correct:
        print "\nFAILURE ON TEST %d" % i
        print "  Output label: %s" % this
        print "  Expected:     %s" % correct
      self.assertEqual(this, correct,
                       msg="Test output label does not match!")
    for i, (this, correct) in enumerate(zip(test_soft, correct_soft)):
      is_equal = self.almostEqual(this, correct)
      if not is_equal:
        print "\nFAILURE ON TEST %d" % i
        print "  Got:      %s" % this
        print "  Expected: %s" % correct
      self.assertTrue(is_equal,
                       msg="Test soft output does not match!")

    self.assertEqual(returncode, 0, msg="Test output does not match!")

class MnistTests(BaseTest):
  def test_minerva(self):
    model_file = "mnist/minerva.conf"
    correct_output = "mnist-minerva.out"
    self.runAndValidate(model_file, correct_output)

  def test_lenet5(self):
    model_file = "mnist/lenet5-ish.conf"
    correct_output = "mnist-lenet5-ish.out"
    self.runAndValidate(model_file, correct_output)

  def test_pruned_csr_minerva(self):
    # The model file was generated with SMIV as the backend (the only one that
    # supports handling sparse data), so it can't be run on the other backends.
    # But the result should be the same as the uncompressed version.
    if ARCH != "smiv":
      return
    model_file = "mnist/minerva.conf"
    correct_output = "mnist-minerva-pruned.out"
    param_file = os.path.join(
        MODEL_DIR, "mnist/trained/smiv/minerva_pruned_csr.txt")
    self.runAndValidate(model_file, correct_output,
                        data_init_mode="READ_FILE", param_file=param_file)

  def test_pruned_no_csr_minerva(self):
    if ARCH != "smiv" or ARCH != "monolithic":
      return
    model_file = "mnist/minerva.conf"
    correct_output = "mnist-minerva-pruned.out"
    param_file = os.path.join(
        MODEL_DIR, "mnist/trained/%s/minerva_pruned.txt" % ARCH)
    self.runAndValidate(model_file, correct_output, data_init_mode="READ_FILE",
                        param_file=param_file)

class GenericTests(BaseTest):
  def test_1_kernel(self):
    model_file = "generic/cnn-1c1k-1p-3fc.conf"
    correct_output = "generic-cnn-1c1k.out"
    self.runAndValidate(model_file, correct_output)

  def test_2_kernels(self):
    model_file = "generic/cnn-1c2k-1p-3fc.conf"
    correct_output = "generic-cnn-1c2k.out"
    self.runAndValidate(model_file, correct_output)

  def test_depthwise_separable(self):
    model_file = "generic/depthwise-separable.conf"
    correct_output = "generic-depthwise-separable.out"
    self.runAndValidate(model_file, correct_output)

  def test_depthwise_act_func(self):
    model_file = "generic/depthwise-act-func.conf"
    correct_output = "generic-depthwise-act-func.out"
    self.runAndValidate(model_file, correct_output)

  def test_strides_2(self):
    model_file = "generic/strides-2.conf"
    correct_output = "generic-strides-2.out"
    self.runAndValidate(model_file, correct_output)

  def test_batch_norm_act_func(self):
    model_file = "generic/batch_norm_act_func.conf"
    correct_output = "generic-batch-norm-act-func.out"
    self.runAndValidate(model_file, correct_output)

class ImageNetTests(BaseTest):
  @unittest.skip("SMIV doesn't support tiling the input image yet.")
  def test_simple(self):
    model_file = "imagenet/simple.conf"
    correct_output = "imagenet-simple.out"
    self.runAndValidate(model_file, correct_output)

  @unittest.skip("VGG takes a really long time and doesn't work with Aladdin yet anyways.")
  def test_vgg16(self):
    model_file = "imagenet/vgg16.conf"
    correct_output = "imagenet-vgg16.out"
    self.runAndValidate(model_file, correct_output)

class SmivTests(BaseTest):
  def test_multi_rounds(self):
    model_file = "smiv/smiv-multi-rounds.conf"
    correct_output = "smiv-multi-rounds.out"
    self.runAndValidate(model_file, correct_output)

  def test_unsupported_act_functions(self):
    model_file = "smiv/unsupported-act-fun.conf"
    correct_output = "smiv-unsupported-act-fun.out"
    self.runAndValidate(model_file, correct_output)

class Cifar10Tests(BaseTest):
  def test_keras_example(self):
    model_file = "cifar10/keras_example.conf"
    correct_output = "cifar10-keras-example.out"
    self.runAndValidate(model_file, correct_output)

  def test_mobilenet(self):
    model_file = "cifar10/mobilenet.conf"
    correct_output = "cifar10-mobilenet.out"
    self.runAndValidate(model_file, correct_output)

class BatchNormTests(BaseTest):
  def test_minerva_bn(self):
    model_file = "mnist/minerva_bn.conf"
    correct_output = "mnist-minerva-bn.out"
    self.runAndValidate(model_file, correct_output)

  def test_conv_bn(self):
    model_file = "generic/conv_bn_test.conf"
    correct_output = "generic-conv-bn.out"
    self.runAndValidate(model_file, correct_output)

class ActivationFuncTests(BaseTest):
  def test_activation_func(self):
    model_file = "mnist/minerva_act_func.conf"
    correct_output = "mnist-minerva-act-func.out"
    self.runAndValidate(model_file, correct_output)

def run_tests():
  suite = unittest.TestSuite()
  test_loader = unittest.TestLoader()
  suite.addTests(test_loader.loadTestsFromName("__main__"))
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  if result.failures or result.errors:
    return 1
  return 0

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("arch", help="Architecture");
  parser.add_argument("binary", help="Path to the executable.")
  args = parser.parse_args()

  global BINARY
  global ARCH
  BINARY = args.binary
  ARCH = args.arch.lower()

  result = run_tests()
  sys.exit(result)

if __name__ == "__main__":
  main()
