#!/usr/bin/env python

import unittest
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import smaug.python.ops.recurrent_test as recurrent_test
from smaug.python.smaug_test import SmaugTest
from smaug.python import global_vars
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.python.ops.array_ops import concat
from smaug.python.ops.attention import BahdanauAttention
from smaug.python.ops.recurrent import LSTM
from smaug.core import types_pb2

class AttentionTest(SmaugTest):
  """Validate SMAUG's attention implementations again TF.

  Attention used in seq2seq models is part of TensorFlow Addons instead of
  TensorFlow core, where an AttentionWrapper is provided to use attention
  in conjunction with an RNN cell.
  """
  def test_bahdanau_attention(self):
    # Build and run an Bahdanau layer in TF.
    batch = 2
    units = 32
    timestep = 8
    # Use the Bahdanau attention mechanism.
    memory = tf.random.normal([batch, timestep, units], dtype=self.dtype)
    attention_mechanism = tfa.seq2seq.BahdanauAttention(
        units=units, memory=memory, dtype=self.dtype)
    # Compute attention using the query and state.
    tf_cell = tf.keras.layers.LSTMCell(
        units, use_bias=False, unit_forget_bias=False, dtype=self.dtype)
    attention_wrapper = tfa.seq2seq.AttentionWrapper(
        tf_cell, attention_mechanism, output_attention=True, dtype=self.dtype)
    query = tf.random.normal([batch, units], dtype=self.dtype)
    tf_initial_state = attention_wrapper.get_initial_state(
        batch_size=batch, dtype=self.dtype)
    # Perform a step of attention-wrapped RNN.
    tf_attention, _ = attention_wrapper(query, tf_initial_state)

    # Build the attention model in SMAUG using the tensors from the TF model.
    query = Tensor(data_layout=types_pb2.NC, tensor_data=query.numpy())
    w, u = recurrent_test.createSmaugWeights(tf_cell)
    memory = Tensor(data_layout=types_pb2.NTC, tensor_data=memory.numpy())
    weights = attention_mechanism.get_weights()
    w_alignment = Tensor(
        data_layout=types_pb2.NC, tensor_data=np.expand_dims(weights[0], 0))
    w_decoder = Tensor(
        data_layout=types_pb2.NC, tensor_data=np.transpose(weights[1]))
    w_encoder = Tensor(
        data_layout=types_pb2.NC, tensor_data=np.transpose(weights[2]))
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      # Create an LSTM and an attention, and perform one step.
      sg_cell = LSTM([w, u])
      sg_attention = BahdanauAttention(memory, w_encoder, w_decoder,
                                       w_alignment)
      sg_initial_attention = Tensor(
          data_layout=types_pb2.NC, tensor_data=np.zeros((batch, units),
                                                         dtype=self.dtype))
      cell_out, _ = sg_cell.step(
          concat([query, sg_initial_attention], axis=1), timestep=0)
      sg_attention(cell_out)
    self.runAndValidate(graph, tf_attention, decimal=2)

if __name__ == "__main__":
  unittest.main()
