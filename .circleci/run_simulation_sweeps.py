import argparse
import os
import subprocess
import six
import shutil
import tempfile

models= ["minerva"]

def run_sim_sweeps(sweep_file, gem5_binary, num_threads):
  sweeps_dir = os.path.join(os.environ["SMAUG_HOME"], "experiments/sweeps")
  sweep_generator = os.path.join(sweeps_dir, "main.py")

  for model in models:
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    process = subprocess.Popen([
        "python", sweep_generator, "--model", model, "--params", sweep_file,
        "--output", temp_dir, "--gem5-binary", gem5_binary, "--run-points",
        "--num-threads",
        str(num_threads)
    ], stdout=None, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    assert process.returncode == 0, (
        "Running the sweeper returned nonzero exit "
        "code Contents of stderr:\n %s" % six.ensure_text(stderr))
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--sweep-file", type=str, help="Sweep generator input file.",
      required=True)
  parser.add_argument(
      "--gem5-binary", type=str, help="gem5 binary file used for simulation.",
      required=True)
  parser.add_argument(
      "--num-threads", type=int,
      help="Number of threads to run the simulations.", default=4)
  args = parser.parse_args()
  run_sim_sweeps(args.sweep_file, args.gem5_binary, args.num_threads)
