import numpy as np

from smaug.core import types_pb2
from smaug.python.tensor import Tensor
from smaug.python.ops import nn_ops
from smaug.python.ops import math_ops
from smaug.python.ops import array_ops
from smaug.python.ops import activation_ops

class LSTM:
  def __init__(
      self, weight_tensors, activation="tanh", activation_params=dict(),
      name="lstm"):
    """ An LSTM layer.

    Args:
      weight_tensors: A list of two weights.
      activation: Activation function used in LSTM.
      activation_params: kwargs for the activation function.
    """
    assert len(weight_tensors) == 2
    self.name = name + ":"
    self.kernel, self.recurrent_kernel = weight_tensors
    self.prepare_states()
    self.activation = activation_ops.get_activation_op(activation)
    self.activation_params = activation_params

  def prepare_states(self):
    """Initialize states as zeros."""
    data_type = self.kernel.tensor_data.dtype
    num_units = (
        self.kernel.shape.dims[0] if self.kernel.shape.layout == types_pb2.NC
        else self.wf.shape.dims[1]) // 4
    self.h = Tensor(
        name=self.name + "/h", data_layout=types_pb2.NC, tensor_data=np.zeros(
            (1, num_units), dtype=data_type))
    self.c = Tensor(
        name=self.name + "/c", data_layout=types_pb2.NC, tensor_data=np.zeros(
            (1, num_units), dtype=data_type))

  def _concat_output_steps(self, outputs):
    outputs_expand = []
    for o in outputs:
      # output is shaped [batch, depth], expand it with the time dimension.
      outputs_expand.append(expand_dims(o, 1, name=self.name + "expand_dims"))
    return concat(outputs_expand, 1, name=self.name + "concat")

  def __call__(self, input_tensor, concat_output=False):
    """Invoke this cell repeatedly until finishing inputs.

    Args:
      input_tensor: Input tensor of shape [batch, time, depth] (aka NTC layout)
        or a series of tensors shaped [batch, depth] (aka NC layout)
        representing timesteps.
      concat_output: If true, the output for each timestep will be concatenated
        into a single tensor, otherwise a list of output tensors will be
        returned.

    Returns:
      Output contains two parts:
      1) Output tensor of shape [batch, time, depth] or
        [batch, depth] * time if not concatenated.
      2) The final state of the LSTM.
    """
    num_steps = 0
    if not isinstance(input_tensor, list):
      input_steps = array_ops.unstack(
          input_tensor, 1, name=self.name + "unstack")
      num_steps = input_tensor.shape.dims[1]
    else:
      input_steps = input_tensor
      num_steps = len(input_steps)
    state = self.c
    output_steps = []
    # Unroll the timesteps.
    for i in range(num_steps):
      output, state = self.step(input_steps[i], i)
      output_steps.append(output)
    if concat_output:
      return self._concat_output_steps(output_steps), state
    return output_steps, state

  def step(self, input_tensor, timestep):
    """Invoke this cell for a single timestep.

    Args:
      input_tensor: An input tensor of shape [batch, depth].
      timestep: The start timestep. This is used for naming the output tensors.

    Returns:
      Output contains two parts:
      1) An output tensor of shape [Batch, Depth].
      2) The final state of the LSTM.
    """
    x = input_tensor
    name_pfx = self.name + "step%d:" % timestep

    z = nn_ops.mat_mul(x, self.kernel, name=name_pfx + "mm_f")
    z = math_ops.add(
        z, nn_ops.mat_mul(self.h, self.recurrent_kernel, name="mm_u"),
        name=name_pfx + "add_z")
    # i = input_gate, c = new_input, f = forget_gate, o = output_gate
    zi, zf, zc, zo = array_ops.split(z, num_or_size_splits=4, axis=1)
    i = activation_ops.sigmoid(zi, name=name_pfx + "sigmoid_i")
    f = activation_ops.sigmoid(zf, name=name_pfx + "sigmoid_f")
    c = math_ops.add(
        math_ops.mul(f, self.c, name=name_pfx + "mul_f"),
        math_ops.mul(
            i,
            self.activation(
                zc, **self.activation_params, name=name_pfx + "act0"),
            name=name_pfx + "mul_i"), name=name_pfx + "add_c")
    o = activation_ops.sigmoid(zo, name=name_pfx + "sigmoid_o")
    h = math_ops.mul(
        o, self.activation(c, **self.activation_params, name=name_pfx + "act1"),
        name=name_pfx + "mul_h")
    self.c = c
    self.h = h
    return self.h, self.c

class BidirectionalLSTM:
  def __init__(
      self, fwd_weight_tensors, bwd_weight_tensors, activation="tanh",
      activation_params=dict(), name="bidir_lstm"):
    """ A bidirectional LSTM layer.

    Args:
      fwd_weight_tensors: weights used for the forward LSTM.
      bwd_weight_tensors: weights used for the backward LSTM.
      activation/activation_params: See in the LSTM class.
    """
    self.name = name + ":"
    self.fwd_lstm = LSTM(
        fwd_weight_tensors,
        activation=activation,
        activation_params=activation_params,
        name=self.name + "fwd_lstm")
    self.bwd_lstm = LSTM(
        bwd_weight_tensors,
        activation=activation,
        activation_params=activation_params,
        name=self.name + "bwd_lstm")

  def __call__(self, input_tensor, concat_output=False):
    """ Invoke the bidirectional LSTM layer.

    Args:
      See in the LSTM class.

    Returns:
      Output contains three parts:
      1) Output tensor of shape [batch, time, depth] or [batch, depth] * time if
        not concatenated.
      2) Final state of the forward LSTM.
      3) Final state of the backward LSTM.
    """
    fwd_outputs, fwd_state = self.fwd_lstm(input_tensor)
    # Reverse the time dimension of the input for the backward LSTM.
    input_bwd = input_tensor
    if isinstance(input_bwd, list):
      input_bwd.reverse()
    else:
      input_bwd = array_ops.unstack(input_bwd, 1)
      input_bwd.reverse()
    bwd_outputs, bwd_state = self.bwd_lstm(input_bwd)
    outputs = []
    for i in range(len(fwd_outputs)):
      outputs.append(
          array_ops.concat([fwd_outputs[i], bwd_outputs[i]], 1,
                           name=self.name + "concat"))
    if concat_output:
      return self._concat_output_steps(outputs), fwd_state, bwd_state
    return outputs, fwd_state, bwd_state
