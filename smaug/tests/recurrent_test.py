#!/usr/bin/env python

import unittest
import tensorflow as tf
import numpy as np

from smaug.tests.tensorflow_test import TensorflowTest
from smaug.python.ops import *
from smaug.python.global_vars import *
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.recurrent import LSTM, BidirectionalLSTM
from smaug.core.types_pb2 import *

class LSTMTest(TensorflowTest):
  def runTest(self):
    # Build and run an LSTM layer in TF.
    tf.keras.backend.set_floatx(backend_datatype[self.backend].__name__)
    inputs = tf.random.normal([1, 8, 32], dtype=backend_datatype[self.backend])
    tf_lstm = tf.keras.layers.LSTM(32, use_bias=False, unit_forget_bias=False)
    tf_output = tf_lstm(inputs)

    # Build the model in SMAUG using the tensors from the TF model.
    w, u = tf_lstm.get_weights()
    input_tensor = Tensor(data_layout=NTC, tensor_data=inputs.numpy())
    w_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(w))
    u_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(u))
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      inputs = input_data(input_tensor)
      sg_lstm = LSTM([w_tensor, u_tensor])
      sg_lstm(input_tensor=inputs)

    self.runAndValidate(graph, tf_output)

class BidirectionalLSTMTest(TensorflowTest):
  def runTest(self):
    # Build and run an BidirectionalLSTM layer in TF.
    tf.keras.backend.set_floatx(backend_datatype[self.backend].__name__)
    inputs = tf.random.normal([1, 8, 32], dtype=backend_datatype[self.backend])
    tf_lstm = tf.keras.layers.LSTM(32, use_bias=False, unit_forget_bias=False)
    tf_bilstm = tf.keras.layers.Bidirectional(layer = tf_lstm)
    tf_output = tf_bilstm(inputs)

    # Build the model in SMAUG using the tensors from the TF model.
    fwd_w, fwd_u, bwd_w, bwd_u = tf_bilstm.get_weights()
    input_tensor = Tensor(data_layout=NTC, tensor_data=inputs.numpy())
    fwd_w_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(fwd_w))
    fwd_u_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(fwd_u))
    bwd_w_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(bwd_w))
    bwd_u_tensor = Tensor(data_layout=NC, tensor_data=np.transpose(bwd_u))
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      inputs = input_data(input_tensor)
      sg_bilstm = BidirectionalLSTM([fwd_w_tensor, fwd_u_tensor],
                                    [bwd_w_tensor, bwd_u_tensor])
      sg_bilstm(input_tensor=inputs)

    self.runAndValidate(graph, tf_output)

if __name__ == "__main__":
  unittest.main()
