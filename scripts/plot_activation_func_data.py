#!/usr/bin/env python
#
# Plot data from the activation function case study.

import argparse
import fnmatch
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_stats_txt(fname):
  stats = ["sim_cycles"]
  data = {}
  with open(fname) as f:
    for line in f:
      split = line.split()
      if not split:
        continue
      for stat_name in stats:
        if stat_name in split[0]:
          print split
          value = float(split[1])
          if stat_name in data:
            data[stat_name]["sim_cycles"] = value
          else:
            data[stat_name] = {"sim_cycles": value}
  return data

def get_total_accel_section(fname):
  """ Parse the simulation trace to find total accelerated section cycles. """
  start_stop = []
  current_data = [0, 0]
  with open(fname, "r") as f:
    for line in f:
      if "Activating accelerator" in line:
        split = line.split(":")
        current_data[0] = int(split[0])
      elif "Accelerator completed" in line:
        split = line.split(":")
        current_data[1] = int(split[0])
        start_stop.append(current_data)
        current_data = [0, 0]

  print start_stop
  cycles = start_stop[-1][1] - start_stop[0][0]
  return cycles

def get_accel_sim_cycles(fname):
  """ Get total cycles executed in the accelerator. """
  sim_data = parse_stats_txt(fname)
  if "sim_cycles" in sim_data:
    return sim_data["sim_cycles"]

def read_sim_data(base_dir):
  data = {}
  # fname = "stats.txt"
  fname = "stdout"
  for root, dirs, files in os.walk(base_dir):
    for item in fnmatch.filter(files, fname):
      exp_name = root.split("/")[-1]
      if exp_name == "cpu":
        continue
      # data[exp_name] = get_accel_sim_cycles(os.path.join(root, item))
      data[exp_name] = get_total_accel_section(os.path.join(root, item))
  return data

def plot_accel_cycles(data):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  xlabels = [k for k in data.iterkeys()]
  data = np.array([data[k] for k in xlabels]).astype(float)
  print data
  data = data/data[-1]
  print data
  print xlabels

  bar_x = np.arange(len(data)) + 0.6
  ax.bar(bar_x, data, width=0.6)
  ax.set_xticklabels(xlabels)
  ax.set_xticks(bar_x)
  ylim = ax.set_ylim(bottom=0.9)
  plt.savefig("total_accel_cycles.pdf", bbox_inches="tight")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("base_dir",
      help="Base directory for all simulation outputs")
  args = parser.parse_args()

  data = read_sim_data(args.base_dir)
  plot_accel_cycles(data)

if __name__ == "__main__":
  main()
