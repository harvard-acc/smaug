#!/usr/bin/env python2
#
# Parses profiling.log files and stores the data into a SQLite3 DB.
#
# Usage:
#   python profile2sql.py sql.db path/to/sim/directory
#
# This script assumes that an experiment identifying string (like dma, acp,
# etc.) is the last directory in the full path to the profiling.log.

import argparse
import csv
import sqlite3
import os
import fnmatch

def process_profiling_log(fname, conn):
  """ Parse the profiling log and store the data into the db. """
  exp_name = fname.split("/")[-2]
  cursor = conn.cursor()
  query = ("insert into profiling_log ("
            "layer_num, layer_type, function, invocation, "
            "start_time, end_time, elapsed_time, desc) "
            "values (?, ?, ?, ?, ?, ?, ?, ?);")

  with open(fname, "rb") as f:
    reader = csv.DictReader(f)
    for row in reader:
      cursor.execute(query, (
          int(row["layer_num"]), row["layer_type"], row["function"],
          int(row["invocation"]), int(row["start_time"]),
          int(row["end_time"]), int(row["elapsed_time"]), exp_name))

  conn.commit()

def create_table_if_not_exists(conn, table_name, query):
  cursor = conn.cursor()
  check_table_query = ("SELECT name FROM sqlite_master "
                       "WHERE type='table' AND name=?;")
  cursor.execute(check_table_query, (table_name,))
  results = cursor.fetchall();
  if len(results) > 0:
    return

  cursor.execute(query)
  conn.commit()

def open_db(db_file):
  conn = sqlite3.connect(db_file)

  # Set up the tables.
  create_table_if_not_exists(conn, "profiling_log",
                             "create table profiling_log ("
                             "layer_num integer, "
                             "layer_type text, "
                             "function text, "
                             "invocation integer, "
                             "start_time integer, "
                             "end_time integer, "
                             "elapsed_time integer, "
                             "desc text);")

  return conn

def read_and_store_profiling_logs(conn, base_dir):
  profiling_file = "profiling.log"
  for root, dirs, files in os.walk(base_dir):
    for item in fnmatch.filter(files, profiling_file):
      fname = os.path.join(root, item)
      process_profiling_log(fname, conn)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("db_file", help="The database file name.")
  parser.add_argument("base_dir",
      help="Search this directory recursively for all profiling logs")
  args = parser.parse_args()

  conn = open_db(args.db_file)
  read_and_store_profiling_logs(conn, args.base_dir)
  conn.close()


if __name__ == "__main__":
  main()
