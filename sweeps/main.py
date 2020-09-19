#!/usr/bin/env python

import argparse
import sys
import os
import json
from sweeper import Sweeper

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="Model name.", required=True)
  parser.add_argument(
      "--params", help="Path to sweep parameter JSON.", required=True)
  parser.add_argument(
      "--output-dir", help="Output directory for generating the data points.",
      required=True)
  parser.add_argument(
      "--gem5-binary", help="Path to the gem5 binary.")
  parser.add_argument(
      "--run-points", action="store_true", default=False,
      help="Option to run the generated data points.")
  parser.add_argument(
      "--num-threads", type=int, default=8,
      help="Number of threads used to run the data points.")
  args = parser.parse_args()

  with open(args.params) as params_file:
    sweep_params = json.load(params_file)

  if not args.gem5_binary:
    args.gem5_binary = os.path.join(
        os.getenv("ALADDIN_HOME"), "../../build/X86/gem5.opt")

  sweeper = Sweeper(args.model, args.output_dir, sweep_params, args.gem5_binary)

  # Start enumerating all the data points.
  sweeper.enumerate_all()

  # Start running simulations for all the generated data points.
  if args.run_points:
    sweeper.run_all(threads=args.num_threads)

if __name__ == "__main__":
  main()
