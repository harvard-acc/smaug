import numpy as np
from types_pb2 import *

smaug_to_np_type = {
    Float16: np.float16,
    Float32: np.float32,
    Float64: np.float64,
    Int32: np.int32,
    Int64: np.int64
}
