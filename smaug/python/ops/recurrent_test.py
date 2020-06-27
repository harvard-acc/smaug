#!/usr/bin/env python

import unittest
import tensorflow as tf
import numpy as np

from smaug.python.smaug_test import SmaugTest
from smaug.python import global_vars
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops.data_op import input_data
from smaug.python.ops.recurrent import LSTM, BidirectionalLSTM
from smaug.core import types_pb2

def createSmaugWeights(tf_lstm):
  """ Extract weights from TF layer and convert them into SMAUG tensors. """
  weights = tf_lstm.get_weights()
  weights_tensors = []
  for w in weights:
    weights_tensors.append(
        Tensor(data_layout=types_pb2.NC, tensor_data=np.transpose(w)))
  return weights_tensors

class LSTMTest(SmaugTest):
  def test_lstm_cell(self):
    # Build and run an LSTM layer in TF.
    tf.keras.backend.set_floatx(
        global_vars.backend_datatype[self.backend].__name__)
    inputs = tf.random.normal([2, 4, 32],
                              dtype=global_vars.backend_datatype[self.backend])
    tf_lstm = tf.keras.layers.LSTM(32, use_bias=False, unit_forget_bias=False)
    tf_output = tf_lstm(inputs)

    # Build the model in SMAUG using the tensors from the TF model.
    inputs_tensor = Tensor(
        data_layout=types_pb2.NTC, tensor_data=inputs.numpy())
    w, u = createSmaugWeights(tf_lstm)
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      inputs = input_data(inputs_tensor)
      sg_lstm = LSTM([w, u])
      sg_lstm(inputs)

    self.runAndValidate(graph, tf_output)

  def test_multilayered_lstm(self):
    # Build and run an LSTM layer in TF.
    tf.keras.backend.set_floatx(
        global_vars.backend_datatype[self.backend].__name__)
    inputs = tf.random.normal([4, 8, 16],
                              dtype=global_vars.backend_datatype[self.backend])

    model = tf.keras.models.Sequential()
    tf_lstm0 = tf.keras.layers.LSTM(
        32, return_sequences=True, use_bias=False, unit_forget_bias=False)
    # We let TF's LSTM only return the last timestep result, because the SMAUG's
    # C++ runtime returns that.
    tf_lstm1 = tf.keras.layers.LSTM(
        32, return_sequences=False, use_bias=False, unit_forget_bias=False)
    model.add(tf_lstm0)
    model.add(tf_lstm1)
    model.compile()
    tf_output = model.predict(inputs)

    # Build the model in SMAUG using the tensors from the TF model.
    inputs_tensor = Tensor(
        data_layout=types_pb2.NTC, tensor_data=inputs.numpy())
    w0, u0 = createSmaugWeights(tf_lstm0)
    w1, u1 = createSmaugWeights(tf_lstm1)
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      inputs = input_data(inputs_tensor)
      sg_lstm0 = LSTM([w0, u0])
      sg_lstm1 = LSTM([w1, u1])
      sg_outputs, state = sg_lstm0(inputs)
      sg_outputs, state = sg_lstm1(sg_outputs)

    self.runAndValidate(graph, tf_output)

  def test_bidirectional_lstm(self):
    # Build and run an BidirectionalLSTM layer in TF.
    tf.keras.backend.set_floatx(
        global_vars.backend_datatype[self.backend].__name__)
    inputs = tf.random.normal([1, 8, 32],
                              dtype=global_vars.backend_datatype[self.backend])
    tf_lstm = tf.keras.layers.LSTM(32, use_bias=False, unit_forget_bias=False)
    tf_bilstm = tf.keras.layers.Bidirectional(layer = tf_lstm)
    tf_output = tf_bilstm(inputs)

    # Build the model in SMAUG using the tensors from the TF model.
    input_tensor = Tensor(data_layout=types_pb2.NTC, tensor_data=inputs.numpy())
    fwd_w, fwd_u, bwd_w, bwd_u = createSmaugWeights(tf_bilstm)
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      inputs = input_data(input_tensor)
      sg_bilstm = BidirectionalLSTM([fwd_w, fwd_u], [bwd_w, bwd_u])
      sg_bilstm(inputs)

    self.runAndValidate(graph, tf_output)

if __name__ == "__main__":
  unittest.main()
