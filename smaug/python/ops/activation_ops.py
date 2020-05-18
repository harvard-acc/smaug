from smaug.core.types_pb2 import *
from smaug.core.node_pb2 import *
from smaug.python.global_vars import *
from smaug.python.ops.common import *

def set_activation_params(activation, act_params_proto, act_params):
  if activation not in supported_activations:
    raise AssertionError(
        ("%s is not a supported activation function. "
         "Supported activations: %s.")
        % (OpType.Name(activation),
           [OpType.Name(a) for a in supported_activations]))
  if act_params != None:
    if activation == LReLU:
      act_params_proto.lrelu_params.CopyFrom(act_params)
    elif activation == ELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == SELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == HardTanh:
      act_params_proto.hard_tanh_params.CopyFrom(act_params)
  else:
    # Use default values for the parameters if not specified.
    if activation == LReLU:
      act_params_proto.lrelu_params.slope = 0.2
    elif activation == ELU:
      act_params_proto.elu_params.alpha = 0.1
    elif activation == SELU:
      act_params_proto.elu_params.alpha = 1.6733
      act_params_proto.elu_params.lambda_param = 1.0507
    elif activation == HardTanh:
      act_params_proto.hard_tanh_params.min = -1
      act_params_proto.hard_tanh_params.max = 1

def activation(op_type):
  """Return an activation function functor.

  Args:
    op_type: OpType of the activation function.
  """
  if op_type == ReLU:
    return relu
  elif op_type == LReLU:
    return lrelu
  elif op_type == ELU:
    return elu
  elif op_type == SELU:
    return selu
  elif op_type == Tanh:
    return tanh
  elif op_type == HardTanh:
    return hard_tanh
  elif op_type == Sigmoid:
    return sigmoid
  elif op_type == Softmax:
    return softmax
  else:
    raise ValueError("The given OpType %s is not an activation function." %
                     OpType.Name(op_type))

def relu(input_tensor, name="relu"):
  return add_node(
      name=name, op=ReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def lrelu(input_tensor, slope=0.2, name="lrelu"):
  params = Params()
  params.act_params.lrelu_params.slope = slope
  return add_node(
      name=name, op=LReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def elu(input_tensor, alpha=0.1, name="relu"):
  params = Params()
  params.act_params.elu_params.alpha = alpha
  return add_node(
      name=name, op=ELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def selu(input_tensor, alpha=1.6733, lambda_param=1.0507, name="selu"):
  params = Params()
  params.act_params.elu_params.alpha = alpha
  params.act_params.elu_params.lambda_param = lambda_param
  return add_node(
      name=name, op=SELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def tanh(input_tensor, name="tanh"):
  return add_node(
      name=name, op=Tanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def hard_tanh(input_tensor, min=-1, max=1, name="hard_tanh"):
  params = Params()
  params.act_params.hard_tanh_params.min = min
  params.act_params.hard_tanh_params.max = max
  return add_node(
      name=name, op=HardTanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def sigmoid(input_tensor, name="sigmoid"):
  return add_node(
      name=name, op=Sigmoid, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def softmax(input_tensor, name=None):
  input_tensor = check_and_add_layout_transform(
      name=name, op=Softmax, input_tensors=[input_tensor])[0]
  return add_node(
      name=name, op=Softmax, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]
