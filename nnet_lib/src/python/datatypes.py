import numpy as np
from types_pb2 import *

np_to_smaug_type = {
    np.float16: Float16,
    np.float32: Float32,
    np.float64: Float64,
    np.int32: Int32,
    np.int64: Int64
}
