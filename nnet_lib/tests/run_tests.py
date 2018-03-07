#!/usr/bin/env python
#
# Integration test runner for nnet_lib.
#
# Usage:
#
#   python run_tests.py arch path/to/executable

import argparse
import math
import sys
import os
import subprocess
import unittest
import tempfile
import shutil
import re

# FP equality checker parameters.
# In theory, each architecture has a different error threshold, due to the
# differences in how they compute the kernels. However, with the low variance of
# the randomly generated weights, the magnitude of the soft targets of the
# output label labels makes custom error thresholds unnecessary.
CHECKERS = {
    "monolithic": {"mode": "fp32", "abs": 0.05, "pct": 0.05},
    "composable": {"mode": "fp32", "abs": 0.05, "pct": 0.05},
    "mkl":        {"mode": "fp32", "abs": 0.05, "pct": 0.05},
    "smiv":       {"mode": "fp32", "abs": 0.05, "pct": 0.05},
    "smv":        {"mode": "fp16", "abs": 0.05, "pct": 0.05},
}

# These get set by the command line argument.
BINARY = ""
ARCH = ""

MODEL_DIR = "test_configs"
CORRECT_OUTPUT_DIR = "correct"
OUTPUT_LABELS = "output_labels.out"

sign = lambda x : math.copysign(1, x)
inf = float("inf")
nan = float("nan")

class FP16:
  # One past the largest magnitude in FP16. This value and all others are
  # rounded to infinity.
  MAX = 65520

  # Represents the maximum integral resolution for a range of values.  Values in
  # the range [0, N) are rounded to the nearest of K, the key.
  ranges = {
      1: (1024, 2048),
      2: (2048, 4096),
      4: (4096, 8192),
      8: (8192, 16384),
      16: (16384, 32768),
      32: (32768, 65520),
  }

  @staticmethod
  def isInRange(value, range_min, range_max):
    return value >= range_min and value < range_max

  @staticmethod
  def getIntegerResolution(value):
    value = abs(value)
    for res, value_range in FP16.ranges.iteritems():
      if FP16.isInRange(value, value_range[0], value_range[1]):
        return res
    return 0

class CheckResult():
  """ Stores the result of an FP comparison. """
  def __init__(self, result, pct_diff, abs_diff):
    """
    Args:
      result: A boolean, indicating whether the check was successful or not.
      pct_diff: The percent difference between the two values being compared.
      abs_diff: The absolute difference between the two values being compared.
    """
    self.result = result
    self.pct_diff = pct_diff
    self.abs_diff = abs_diff

class EqualityChecker():
  """ A functor to compare if two FP values are approximately equal. """
  def __init__(self, mode, fp_err_abs, fp_err_pct):
    """ Construct a EqualityChecker object.

    Two values are considered approximately equal if either of the two error
    distances is within their respective margins. For FP16 values, we also
    consider the integer representation resolution limit for large values.

    Args:
      mode: fp16 or fp32. fp32 has more strict thresholds than fp16.
      fp_err_abs: The absolute error threshold.
      fp_err_pct: The percent error threshold.
    """
    self.mode = mode
    self.fp_err_abs = fp_err_abs
    self.fp_err_pct = fp_err_pct

  def __call__(self, values, reference):
    compare_func = (self.compareFP32Scalar if self.mode == "fp32"
                    else self.compareFP16Scalar)
    return self.almostEqual(values, reference, compare_func)

  def getAbsErr(self, value, reference):
    return abs(reference - value)

  def getPctErr(self, value, reference):
    if reference == 0:
      return 0 if value == 0 else inf
    return self.getAbsErr(value, reference) / abs(reference) * 100

  def compareFP32Scalar(self, value, reference):
    abs_diff = self.getAbsErr(value, reference)
    pct_diff = self.getPctErr(value, reference)
    if sign(value) != sign(reference):
      return CheckResult(False, abs_diff, pct_diff)
    result = (pct_diff < self.fp_err_pct or
              abs_diff < self.fp_err_abs)
    return CheckResult(result, pct_diff, abs_diff)

  def compareFP16Scalar(self, value, reference):
    """ Compare two values with the FP16 standard.

    In addition to comparing if the values are within their error margins, we
    also define approximately equal if:
      1) @value is inf and @reference is greater than FP16.MAX (abs value).
      2) the absolute difference is within the range of the integral
         representation resolution.
    """
    abs_diff = self.getAbsErr(value, reference)
    pct_diff = self.getPctErr(value, reference)
    if sign(value) != sign(reference):
      return CheckResult(False, 0, 0)
    if abs(value) == inf and abs(reference) >= FP16.MAX:
      return CheckResult(True, 0, 0)
    int_res = FP16.getIntegerResolution(value)
    result = (pct_diff < self.fp_err_pct or
              abs_diff < max(self.fp_err_abs, int_res))
    return CheckResult(result, pct_diff, abs_diff)

  def almostEqual(self, values, references, compare_func):
    """ Returns true if val and ref are approximately equal.

    val and ref are list-like objects.
    """
    results = [compare_func(v, r) for v, r in zip(values, references)]
    all_pass = all(res.result for res in results)
    if all_pass:
      return True
    failing_tests = [res for res in results if not res]

    print ""
    print "% error   : ", ", ".join(
        ["{:10.4f}".format(res.pct_diff) for res in results])
    print "abs error : ", ", ".join(
        ["{:10.4f}".format(res.abs_diff) for res in results])
    return False

class BaseTest(unittest.TestCase):
  def setUp(self):
    self.run_dir = tempfile.mkdtemp()
    self.output_filename = os.path.join(self.run_dir, "stdout")
    checker_args = CHECKERS[ARCH]
    self.checker = EqualityChecker(
        checker_args["mode"], checker_args["abs"], checker_args["pct"])

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
          values = [float(v.strip()) for v in
                    line.replace("[", "").replace("]", "").split()]
          test_soft.append(values)
    return test_pred, test_soft

  def launchSubprocess(self, cmd):
    with open(self.output_filename, "w") as f:
      fnull = open(os.devnull, "w")
      returncode = subprocess.call(cmd, shell=True, stdout=fnull, stderr=f)
      fnull.close()

    if returncode != 0:
      print "\nTEST FAILED! Contents of stderr:"
      print "--------------------------------\n"
      with open(self.output_filename, "r") as f:
        for line in f:
          print line.strip()
      print "--------------------------------"

    return returncode

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
      is_equal = self.checker(this, correct)
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
    if not ARCH in ["smiv", "smv"]:
      return
    model_file = "mnist/minerva.conf"
    correct_output = "mnist-minerva-pruned.out"
    param_file = os.path.join(
        MODEL_DIR, "mnist/trained/%s/minerva_pruned_csr.txt" % ARCH)
    self.runAndValidate(model_file, correct_output,
                        data_init_mode="READ_FILE", param_file=param_file)

  def test_pruned_no_csr_minerva(self):
    if ARCH == "composable":
      return
    model_file = "mnist/minerva.conf"
    correct_output = "mnist-minerva-pruned.out"
    param_file = os.path.join(
        MODEL_DIR, "mnist/trained/%s/minerva_pruned.txt" % ARCH)
    self.runAndValidate(model_file, correct_output, data_init_mode="READ_FILE",
                        param_file=param_file)

class MinervaAccessMechanismTests(BaseTest):
  """ These tests exercise different combinations of the offload mechanisms. """
  def setUp(self):
    """ All these tests should produce the SAME output. """
    super(MinervaAccessMechanismTests, self).setUp()
    self.correct_output = "mnist-minerva.out"

  def runAndValidate(self, model_file, correct_output):
    super(MinervaAccessMechanismTests, self).runAndValidate(
        model_file, correct_output)

  def test_minerva_all_cache(self):
    model_file = "mnist/minerva-access-mechs/minerva_cache.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_minerva_all_acp(self):
    model_file = "mnist/minerva-access-mechs/minerva_acp.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_minerva_dma_acp(self):
    model_file = "mnist/minerva-access-mechs/minerva_dma_acp.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_minerva_dma_cache(self):
    model_file = "mnist/minerva-access-mechs/minerva_dma_cache.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_minerva_dma_acp_no_hw_act_func(self):
    model_file = "mnist/minerva-access-mechs/minerva_dma_acp_no_hw_act_func.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_minerva_dma_cache_no_hw_act_func(self):
    model_file = "mnist/minerva-access-mechs/minerva_dma_cache_no_hw_act_func.conf"
    self.runAndValidate(model_file, self.correct_output)

class MinervaAccessMechanismCsrTests(MinervaAccessMechanismTests):
  """ Same as MinervaAccessMechanismTests, but with compressed weights. """
  def setUp(self):
    super(MinervaAccessMechanismTests, self).setUp()
    self.correct_output = "mnist-minerva-pruned.out"
    self.param_file = os.path.join(
        MODEL_DIR, "mnist/trained/%s/minerva_pruned.txt" % ARCH)

  def runAndValidate(self, model_file, correct_output):
    """ Supply the model parameter file as an additional argument. """
    if not ARCH in ["smiv", "smv"]:
      return
    super(MinervaAccessMechanismTests, self).runAndValidate(
        model_file, correct_output,
        param_file=self.param_file, data_init_mode="READ_FILE");

class Cifar10CnnAccessMechanismTests(BaseTest):
  """ These tests exercise different combinations of the offload mechanisms. """
  def setUp(self):
    """ All these tests should produce the SAME output from random data. """
    super(Cifar10CnnAccessMechanismTests, self).setUp()
    self.correct_output = "cifar10-keras-example-random-data.out"

  def test_cifar10_cnn_all_cache(self):
    model_file = "cifar10/cnn-access-mechs/cnn-cache.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_all_acp(self):
    model_file = "cifar10/cnn-access-mechs/cnn-acp.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_dma_acp(self):
    model_file = "cifar10/cnn-access-mechs/cnn-dma-acp.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_dma_cache(self):
    model_file = "cifar10/cnn-access-mechs/cnn-dma-cache.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_acp_no_hw_act_func(self):
    model_file = "cifar10/cnn-access-mechs/cnn-acp-no-hw-act-func.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_cache_no_hw_act_func(self):
    model_file = "cifar10/cnn-access-mechs/cnn-cache-no-hw-act-func.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_dma_acp_no_hw_act_func(self):
    model_file = "cifar10/cnn-access-mechs/cnn-dma-acp-no-hw-act-func.conf"
    self.runAndValidate(model_file, self.correct_output)

  def test_cifar10_cnn_dma_cache_no_hw_act_func(self):
    model_file = "cifar10/cnn-access-mechs/cnn-dma-cache-no-hw-act-func.conf"
    self.runAndValidate(model_file, self.correct_output)

class Cifar10RealDataCnnAccessMechanismTests(Cifar10CnnAccessMechanismTests):
  def setUp(self):
    super(Cifar10RealDataCnnAccessMechanismTests, self).setUp()
    self.correct_output = "cifar10-keras-example-real-data.out"
    self.param_file = os.path.join(
        MODEL_DIR, "cifar10/trained/%s/cnn-pruned.txt" % ARCH)

  def runAndValidate(self, model_file, correct_output):
    """ Supply the model parameter file as an additional argument. """
    super(Cifar10RealDataCnnAccessMechanismTests, self).runAndValidate(
        model_file, correct_output,
        param_file=self.param_file, data_init_mode="READ_FILE");

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
  def test_keras_example_random_data(self):
    model_file = "cifar10/keras_example.conf"
    correct_output = "cifar10-keras-example-random-data.out"
    self.runAndValidate(model_file, correct_output)

  def test_keras_example_real_data(self):
    if ARCH == "composable":
      return
    model_file = "cifar10/keras_example.conf"
    correct_output = "cifar10-keras-example-real-data.out"
    param_file = os.path.join(
        MODEL_DIR, "cifar10/trained/%s/cnn-pruned.txt" % ARCH)
    self.runAndValidate(model_file, correct_output,
          data_init_mode="READ_FILE", param_file=param_file)

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
  def test_mnist_minerva_activation_func(self):
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
