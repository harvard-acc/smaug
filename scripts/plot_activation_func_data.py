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

def read_sim_data(base_dir):
  data = {}
  for root, dirs, files in os.walk(base_dir):
    for item in fnmatch.filter(files, "stats.txt"):
      exp_name = root.split("/")[-2]
      sim_data = parse_stats_txt(os.path.join(root, item))
      if "sim_cycles" in sim_data:
        data[exp_name] = sim_data["sim_cycles"]
  return data

def plot_accel_cycles(data):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  xlabels = [k for k in data.iterkeys()]
  data = [data[k]["sim_cycles"] for k in xlabels]

  bar_x = np.arange(len(data)) + 0.3
  ax.bar(bar_x, data, width=0.6)
  ax.set_xticklabels(xlabels)
  ax.set_xticks(bar_x)
  ylim = ax.set_ylim(bottom=0)
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
