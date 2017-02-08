#!/usr/bin/env python
#
# Concatenate all given file names into a common file.

import os
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("srcs", help="All *.c source files.")
  parser.add_argument("output", help="Output file path.")
  args = parser.parse_args()

  srcs = args.srcs.split()
  with open(args.output, "w") as f:
    for src in srcs:
      f.write("#include \"%s\"\n" % src)

if __name__ == "__main__":
  main()
