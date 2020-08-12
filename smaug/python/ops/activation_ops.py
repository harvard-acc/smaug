from smaug.core import types_pb2
from smaug.core import node_pb2
from smaug.python import global_vars
from smaug.python.ops import common

def _set_activation_params(activation, params, proto):
  """Set the parameters of the activation function.

  Args:
    activation: An activation op type such as `types_pb2.ReLU`.
    params: kwargs for the activation function parameters.
    proto: An `ActivationParams`, the proto to set.
  """
  if params is not None:
    if activation == types_pb2.LReLU:
      proto.lrelu_params.slope = params["slope"]
    elif activation == types_pb2.ELU:
      proto.elu_params.alpha = params["alpha"]
    elif activation == types_pb2.SELU:
      proto.elu_params.alpha = params["alpha"]
      proto.elu_params.lambda_param = params["lambda_param"]
    elif activation == types_pb2.HardTanh:
      proto.hard_tanh_params.min = params["min"]
      proto.hard_tanh_params.max = params["max"]
  else:
    # Use default values for the parameters if not specified.
    if activation == types_pb2.LReLU:
      proto.lrelu_params.slope = 0.2
    elif activation == types_pb2.ELU:
      proto.elu_params.alpha = 0.1
    elif activation == types_pb2.SELU:
      proto.elu_params.alpha = 1.6733
      proto.elu_params.lambda_param = 1.0507
    elif activation == types_pb2.HardTanh:
      proto.hard_tanh_params.min = -1
      proto.hard_tanh_params.max = 1

def get_activation_op(activation):
  """Return an activation function functor.

  Args:
    activation: A string representing the activation function.
  """
  return _activation_type_op_tuples[activation][1]

def to_proto(activation, params):
  """Return the activation proto.

  Args:
    activation: A string representing the activation function.
    params: kwargs for the activation function parameters.
    proto: An `ActivationParams`, the proto to set.

  Returns:
    An `ActivationParams`, the proto.
  """
  proto = node_pb2.ActivationParams()
  act_type = _activation_type_op_tuples[activation][0]
  proto.activation = act_type
  _set_activation_params(act_type, params, proto)
  return proto

def relu(input_tensor, name="relu"):
  """Rectified linear unit operator."""
  return common.add_node(
      name=name, op=types_pb2.ReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def lrelu(input_tensor, slope=0.2, name="lrelu"):
  """Leaky rectified linear unit operator: max(slope * x, 0)."""
  params = node_pb2.Params()
  params.act_params.lrelu_params.slope = slope
  return common.add_node(
      name=name, op=types_pb2.LReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def elu(input_tensor, alpha=0.1, name="relu"):
  """Exponential linear unit function.

  Defined as:
    if input_tensor > 0, alpha * exp(input_tensor - 1), else input_tensor.
  """
  params = node_pb2.Params()
  params.act_params.elu_params.alpha = alpha
  return common.add_node(
      name=name, op=types_pb2.ELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def selu(input_tensor, alpha=1.6733, lambda_param=1.0507, name="selu"):
  """Scaled exponential linear unit function.

  Defined as: lambda_param * elu(input_tensor, alpha).
  """
  params = node_pb2.Params()
  params.act_params.elu_params.alpha = alpha
  params.act_params.elu_params.lambda_param = lambda_param
  return common.add_node(
      name=name, op=types_pb2.SELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def tanh(input_tensor, name="tanh"):
  """Tanh operator."""
  return common.add_node(
      name=name, op=types_pb2.Tanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def hard_tanh(input_tensor, min=-1, max=1, name="hard_tanh"):
  """Hard tanh operator.

  This bounds the min and max values of the tanh operator.
  """
  params = node_pb2.Params()
  params.act_params.hard_tanh_params.min = min
  params.act_params.hard_tanh_params.max = max
  return common.add_node(
      name=name, op=types_pb2.HardTanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def sigmoid(input_tensor, name="sigmoid"):
  """Sigmoid operator.

  Defined as 1/(1 + exp(-input_tensor)).
  """
  return common.add_node(
      name=name, op=types_pb2.Sigmoid, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def softmax(input_tensor, name=None):
  """Softmax operator."""
  input_tensor = common.check_and_add_layout_transform(
      name=name, op=types_pb2.Softmax, input_tensors=[input_tensor])[0]
  return common.add_node(
      name=name, op=types_pb2.Softmax, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

_activation_type_op_tuples = {
    "relu": (types_pb2.ReLU, relu),
    "lrelu": (types_pb2.LReLU, lrelu),
    "elu": (types_pb2.ELU, elu),
    "selu": (types_pb2.SELU, selu),
    "tanh": (types_pb2.Tanh, tanh),
    "hard_tanh": (types_pb2.HardTanh, hard_tanh),
    "sigmoid": (types_pb2.Sigmoid, sigmoid),
    "softmax": (types_pb2.Softmax, softmax)
}
