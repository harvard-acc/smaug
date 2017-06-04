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

# This gets set by the command line argument.
BINARY = ""

# Configuration file, correct output file.
TESTS = [
    ("generic/cnn-1c1k-1p-3fc.conf", "generic-cnn-1c1k.out"),
    ("generic/cnn-1c2k-1p-3fc.conf", "generic-cnn-1c2k.out"),
    ("mnist/minerva.conf",           "mnist-minerva.out"),
    ("mnist/lenet5-ish.conf",        "mnist-lenet5-ish.out"),
    ("imagenet/simple.conf",         "imagenet-simple.out"),
    ("imagenet/vgg16.conf",          "imagenet-vgg16.out"),
]

MODEL_DIR = "../../models/"
CORRECT_OUTPUT_DIR = "correct"

class BaseTest(unittest.TestCase):
  def setUp(self):
    self.run_dir = tempfile.mkdtemp()
    self.output_filename = os.path.join(self.run_dir, "stdout")

  def tearDown(self):
    shutil.rmtree(self.run_dir)
    os.remove("output_labels.out")

  def launchSubprocess(self, cmd):
    with open(self.output_filename, "w") as f:
      returncode = subprocess.call(
          cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

    if returncode != 0:
      print "\nTEST FAILED! Contents of stdout:"
      print "--------------------------------\n"
      with open(self.output_filename, "r") as f:
        for line in f:
          print line
      print "--------------------------------"

    return returncode

  def runAndValidate(self, model_file, correct_output):
    returncode = self.launchSubprocess(
          "%s %s" % (BINARY, os.path.join(MODEL_DIR, model_file)))

    self.assertEqual(returncode, 0, msg="Test returned nonzero exit code!")

    returncode = self.launchSubprocess(
          "diff output_labels.out %s" % os.path.join(
              CORRECT_OUTPUT_DIR, correct_output))

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

class GenericTests(BaseTest):
  def test_1_kernel(self):
    model_file = "generic/cnn-1c1k-1p-3fc.conf"
    correct_output = "generic-cnn-1c1k.out"
    self.runAndValidate(model_file, correct_output)

  def test_2_kernels(self):
    model_file = "generic/cnn-1c2k-1p-3fc.conf"
    correct_output = "generic-cnn-1c2k.out"
    self.runAndValidate(model_file, correct_output)

class ImageNetTests(BaseTest):
  def test_simple(self):
    model_file = "imagenet/simple.conf"
    correct_output = "imagenet-simple.out"
    self.runAndValidate(model_file, correct_output)

  @unittest.skip("VGG takes a really long time and doesn't work with Aladdin yet anyways.")
  def test_vgg16(self):
    model_file = "imagenet/vgg16.conf"
    correct_output = "imagenet-vgg16.out"
    self.runAndValidate(model_file, correct_output)

def run_tests():
  suite = unittest.TestSuite()
  test_loader = unittest.TestLoader()
  suite.addTests(test_loader.loadTestsFromName("__main__"))
  result = unittest.TextTestRunner(verbosity=2).run(suite)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("binary", help="Path to the executable.")
  args = parser.parse_args()

  global BINARY
  BINARY = args.binary

  run_tests()

if __name__ == "__main__":
  main()
