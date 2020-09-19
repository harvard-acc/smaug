import os
from configparser import RawConfigParser
import fileinput

def change_config_file(point_dir, config_file, kv_map):
  f = fileinput.input(os.path.join(point_dir, config_file), inplace=True)
  for line in f:
    for k in kv_map:
      if k in line:
        line = line % kv_map
    print(line, end="")
  f.close()

class BaseParam:
  def __init__(self, name, sweep_vals, changes_trace=False):
    self._name = name
    self._sweep_vals = sweep_vals
    self._changes_trace = changes_trace
    self._curr_sweep_idx = -1

  def __str__(self):
    return "%s_%s" % (self._name, str(self.curr_sweep_value()))

  @property
  def changes_trace(self):
    return self._changes_trace

  def curr_sweep_value(self):
    return self._sweep_vals[self._curr_sweep_idx]

  def apply(self, point_dir):
    raise NotImplementedError

  @classmethod
  def default_value(cls):
    raise NotImplementedError

  def next(self):
    self._curr_sweep_idx += 1
    if self._curr_sweep_idx == len(self._sweep_vals):
      self._curr_sweep_idx = -1
      return False
    return True

class NumThreadsParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {
            "num-threads": self.curr_sweep_value(),
            "num-cpus": self.curr_sweep_value() + 1
        })

  @classmethod
  def default_value(cls):
    return [1]

class NumAccelsParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, True)

  def apply(self, point_dir):
    # Change the run.sh, gem5.cfg and trace.sh.
    change_config_file(
        point_dir, "run.sh", {"num-accels": self.curr_sweep_value()})
    if self._sweep_vals[self._curr_sweep_idx] > 1:
      self._change_gem5_cfg(point_dir)
    change_config_file(
        point_dir, "trace.sh", {"num-accels": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [1]

  def _change_gem5_cfg(self, point_dir):
    gem5cfg = RawConfigParser()
    gem5cfg_file = os.path.join(point_dir, "gem5.cfg")
    gem5cfg.read(gem5cfg_file)
    acc0 = gem5cfg.sections()[0]
    acc0_id = int(gem5cfg.get(acc0, "accelerator_id"))
    with open(gem5cfg_file, "w") as cfg:
      for n in range(1, self.curr_sweep_value()):
        new_acc = "acc" + str(n)
        gem5cfg.add_section(new_acc)
        for key, value in gem5cfg.items(acc0):
          gem5cfg.set(new_acc, key, value)
          gem5cfg.set(new_acc, "accelerator_id", str(acc0_id + n))
          gem5cfg.set(
              new_acc, "trace_file_name", "./dynamic_trace_acc%d.gz" % n)
      gem5cfg.write(cfg)

class SoCInterfaceParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "model_files", {"soc_interface": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return ["dma"]

class L1DSizeParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l1d_size": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    # 32KB by default.
    return [32768]

class L2SizeParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l2_size": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    # 4MB by default.
    return [4194304]

class L1DAssocParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l1d_assoc": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [4]

class L2AssocParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l2_assoc": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [16]

class L1DHitLatencyParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l1d_hit_latency": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [1]

class L2HitLatencyParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"l2_hit_latency": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [5]

class AccClockParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh",
        {"sys-clock": "%.3fGHz" % (1.0 / self.curr_sweep_value())})
    change_config_file(
        point_dir, "smv-accel.cfg", {"cycle_time": self.curr_sweep_value()})
    change_config_file(
        point_dir, "gem5.cfg", {"cycle_time": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [1]

class CpuClockParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh",
        {"cpu-clock": "%.3fGHz" % (1.0 / self.curr_sweep_value())})

  @classmethod
  def default_value(cls):
    return [0.5]

class MemTypeParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "run.sh", {"mem-type": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return ["LPDDR3_1600_1x32"]

class PipelinedDmaParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg", {"pipelined_dma": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [1]

class IgnoreCacheFlushParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg", {"ignore_cache_flush": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [0]

class InvalidateOnDmaStoreParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg",
        {"invalidate_on_dma_store": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [0]

class MaxDmaRequestsParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg", {"max_dma_requests": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [16]

class NumDmaChannelsParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg", {"num_dma_channels": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [1]

class DmaChunkSizeParam(BaseParam):
  def __init__(self, name, sweep_vals):
    BaseParam.__init__(self, name, sweep_vals, False)

  def apply(self, point_dir):
    change_config_file(
        point_dir, "gem5.cfg", {"dma_chunk_size": self.curr_sweep_value()})

  @classmethod
  def default_value(cls):
    return [64]
