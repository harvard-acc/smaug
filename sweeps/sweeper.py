import os
import sys
import errno
import six
import shutil
import subprocess
import multiprocessing as mp
from params import *

param_types = {
    "num_threads": NumThreadsParam,
    "num_accels": NumAccelsParam,
    "soc_interface": SoCInterfaceParam,
    "l1d_size": L1DSizeParam,
    "l2_size": L2SizeParam,
    "l1d_assoc": L1DAssocParam,
    "l2_assoc": L2AssocParam,
    "l1d_hit_latency": L1DHitLatencyParam,
    "l2_hit_latency": L2HitLatencyParam,
    "acc_clock": AccClockParam,
    "mem_type": MemTypeParam,
    "cpu_clock": CpuClockParam,
    "pipelined_dma": PipelinedDmaParam,
    "ignore_cache_flush": IgnoreCacheFlushParam,
    "invalidate_on_dma_store": InvalidateOnDmaStoreParam,
    "max_dma_requests": MaxDmaRequestsParam,
    "num_dma_channels": NumDmaChannelsParam,
    "dma_chunk_size": DmaChunkSizeParam,
}

class Sweeper:
  def __init__(self, model_name, output_dir, params, gem5_binary):
    self._model_name = model_name
    self._output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(self._output_dir):
      os.mkdir(self._output_dir)
    self._configs_dir = os.path.join(os.getenv("SMAUG_HOME"), "sweeps/configs")
    self._init_params(params)
    self._gem5_binary = gem5_binary
    self._num_data_points = 0
    self._traces = set()
    # Create a folder for storing all the traces.
    self._trace_dir = os.path.join(self._output_dir, "traces")
    if not os.path.isdir(self._trace_dir):
      os.mkdir(self._trace_dir)

  def _init_params(self, params):
    self._params = []
    for param_name,param_type in param_types.items():
      if param_name in params:
        self._params.append(param_type(param_name, params[param_name]))
      else:
        self._params.append(param_type(param_name, param_type.default_value()))

  def curr_point_dir(self):
    return os.path.join(self._output_dir, str(self._num_data_points))

  def _create_point(self):
    point_dir = os.path.join(self._output_dir, str(self._num_data_points))

    # Copy configuration files to the simulation directory of this data point.
    if not os.path.isdir(point_dir):
      os.mkdir(point_dir)
    for f in ["gem5.cfg", "run.sh", "model_files", "smv-accel.cfg", "trace.sh"]:
      shutil.copyfile(
          os.path.join(self._configs_dir, f),
          os.path.join(point_dir, f))
    for f in ["env.txt"]:
      link = os.path.join(point_dir, f)
      target = os.path.join(self._configs_dir, f)
      try:
        os.symlink(target, link)
      except OSError as e:
        if e.errno == errno.EEXIST:
          os.remove(link)
          os.symlink(target, link)
        else:
          raise e
    soc_interface = "dma"
    for p in self._params:
      if isinstance(p, SoCInterfaceParam):
        soc_interface = p.curr_sweep_value()
    change_config_file(
        point_dir, "model_files", {
            "model_name": self._model_name,
            "soc_interface": soc_interface
        })
    # gem5 binary in run.sh.
    change_config_file(point_dir, "run.sh", {"gem5-binary": self._gem5_binary})

    # Apply every sweep parameter for this data point.
    for p in self._params:
      p.apply(self.curr_point_dir())

    # Now all the configuration files have been updated, Check if we need to
    # generate new trace for this data point.
    trace_id = None
    num_accels = 0
    for p in self._params:
      if p.changes_trace == True:
        trace_id = "%s_%s" % (trace_id, param) if trace_id else str(p)
      if isinstance(p, NumAccelsParam):
        num_accels = p.curr_sweep_value()
    # Before we generate any traces, create links to the traces.
    for i in range(num_accels):
      link = os.path.join(point_dir, "dynamic_trace_acc%d.gz" % i)
      target = os.path.join(
          self._trace_dir, trace_id, "dynamic_trace_acc%d.gz" % i)
      try:
        os.symlink(target, link)
      except OSError as e:
        if e.errno == errno.EEXIST:
          os.remove(link)
          os.symlink(target, link)
        else:
          raise e
    # If this is a new trace id, generate new traces.
    if trace_id not in self._traces:
      self._traces.add(trace_id)
      trace_dir = os.path.join(self._trace_dir, trace_id)
      if not os.path.isdir(trace_dir):
        os.mkdir(trace_dir)
      # Run trace.sh to generate the traces.
      process = subprocess.Popen(["bash", "trace.sh"], cwd=point_dir,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = process.communicate()
      assert process.returncode == 0, (
          "Generating trace returned nonzero exit code! Contents of output:\n "
          "%s\n%s" % (six.ensure_text(stdout), six.ensure_text(stderr)))
    print("---Created data point: %d.---" % self._num_data_points)

  def enumerate(self, param_idx):
    if param_idx < len(self._params) - 1:
      while self._params[param_idx].next() == True:
        self.enumerate(param_idx + 1)
      return
    else:
      while self._params[param_idx].next() == True:
        self._create_point()
        self._num_data_points += 1
      return

  def enumerate_all(self):
    """Create configurations for all data points.  """
    print("Creating all data points...")
    self.enumerate(0)

  def run_all(self, threads):
    """Run simulations for all data points.

    Args:
      Number of threads used to run the simulations.
    """
    print("Running all data points...")
    counter = mp.Value('i', 0)
    sims = []
    pool = mp.Pool(
        initializer=_init_counter, initargs=(counter, ), processes=threads)
    for p in range(self._num_data_points):
      cmd = os.path.join(self._output_dir, str(p), "run.sh")
      sims.append(pool.apply_async(_run_simulation, args=(cmd, )))
    for sim in sims:
      sim.get()
    pool.close()
    pool.join()

counter = 0

def _init_counter(args):
  global counter
  counter = args

def _run_simulation(cmd):
  global counter
  process = subprocess.Popen(["bash", cmd], cwd=os.path.dirname(cmd),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  assert process.returncode == 0, (
      "Running simulation returned nonzero exit code! Contents of output:\n "
      "%s\n%s" % (six.ensure_text(stdout), six.ensure_text(stderr)))
  with counter.get_lock():
    counter.value += 1
  print("---Finished running points: %d.---" % counter.value)
