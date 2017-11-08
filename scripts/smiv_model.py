#!/usr/bin/env python
#
# A basic analytical performance model for the SMIV accelerator blocks.
#
# Use this script to estimate how many cycles it should take for an accelerator
# block to finish. When debugging gem5-aladdin performance bugs, this is useful
# to see what our target area is. Note that this assumes no stalls on data
# beyond what is explicitly accounted for!
#
# Note: for reduction, use the input rows/cols pre convolution, and supply the
# kernel size; the script will calculate the correct number of output rows/cols.
#
# This supports convolution with stride 1 and kernel size > 1 only!

import argparse
import math

def run_conv_model(args):
  k_size = args.kernel_size
  if k_size == 1:
    assert("1x1 convolution not supported yet!")
  i_rows = args.input_rows
  i_cols = args.input_cols

  total_cycles = 0
  cols = i_cols - k_size + 1
  while cols > 0:
    cycles = 0
    if cols >= 8:
      cycles += 4 if k_size <= 4 else 8
      cols -= 8
    else:
      cycles += math.ceil(cols/2.0) if k_size <= 4 else cols
      cols -= cols
    cycles += 1  # for activation load and psum merge
    cycles *= k_size  # Repeat this many times for each output row
    cycles *= (i_rows - k_size + 1)  # For all output rows
    total_cycles += cycles

  total_cycles *= args.input_chans  # For each channel
  # cycles *= args.num_kerns  # For each kernel
  print "Estimated cycles for convolution: %d" % total_cycles

def run_reduce_model(args):
  k_size = args.kernel_size
  i_rows = args.input_rows
  i_cols = args.input_cols
  i_chans = args.input_chans

  o_rows = i_rows - k_size + 1
  o_cols = i_cols - k_size + 1

  # + 1 for activation func. scratchpad load/store are pipelined.
  cycles_to_reduce_all_chans = i_chans + 1
  col_iters = math.ceil(o_cols / 8.0)

  total_cycles = cycles_to_reduce_all_chans * col_iters * o_rows
  print "Estimated cycles for reduction: %d" % total_cycles

def run_fc_model(args):
  w_rows = args.input_rows
  w_cols = args.input_cols
  i_batch = args.batch_size

  # Bias is preloaded as part of the input load, so this is fully pipelined.
  row_iters = math.ceil(w_rows/8.0)
  row_iter_cycles = w_cols * i_batch
  row_iter_cycles += i_batch  # For activation fun per batch.
  total_cycles = row_iters * row_iter_cycles

  print "Estimated cycles for FC: %d" % total_cycles

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("block", choices=["conv", "fc", "reduce"])
  parser.add_argument("-r", "--input-rows", type=int,
      help="Rows in the input (except for FC, where it is rows of weights")
  parser.add_argument("-c", "--input-cols", type=int,
      help="Cols in the input (except for FC, where it is cols of weights")
  parser.add_argument("-a", "--input-chans", type=int,
      help="Input channels (CONV and Reduce only)")
  parser.add_argument("-k", "--kernel-size", type=int,
      help="Kernel size (CONV and reduce only")
  parser.add_argument("-b", "--batch-size", type=int, default=1,
      help="FC only")

  args = parser.parse_args()

  if args.block == "conv":
    run_conv_model(args)
  elif args.block == "reduce":
    run_reduce_model(args)
  elif args.block == "fc":
    run_fc_model(args)
  else:
    assert("Invalid block!")


if __name__ == "__main__":
  main()
