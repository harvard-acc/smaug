from smaug.core import types_pb2
from smaug.core import node_pb2
from smaug.python import global_vars
from smaug.python.ops import common

def set_activation_params(activation, act_params_proto, act_params):
  if activation not in global_vars.supported_activations:
    raise AssertionError(
        ("%s is not a supported activation function. "
         "Supported activations: %s.")
        % (OpType.Name(activation),
           [OpType.Name(a) for a in supported_activations]))
  if act_params != None:
    if activation == types_pb2.LReLU:
      act_params_proto.lrelu_params.CopyFrom(act_params)
    elif activation == types_pb2.ELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == types_pb2.SELU:
      act_params_proto.elu_params.CopyFrom(act_params)
    elif activation == types_pb2.HardTanh:
      act_params_proto.hard_tanh_params.CopyFrom(act_params)
  else:
    # Use default values for the parameters if not specified.
    if activation == types_pb2.LReLU:
      act_params_proto.lrelu_params.slope = 0.2
    elif activation == types_pb2.ELU:
      act_params_proto.elu_params.alpha = 0.1
    elif activation == types_pb2.SELU:
      act_params_proto.elu_params.alpha = 1.6733
      act_params_proto.elu_params.lambda_param = 1.0507
    elif activation == types_pb2.HardTanh:
      act_params_proto.hard_tanh_params.min = -1
      act_params_proto.hard_tanh_params.max = 1

def activation(op_type):
  """Return an activation function functor.

  Args:
    op_type: OpType of the activation function.
  """
  if op_type == types_pb2.ReLU:
    return relu
  elif op_type == types_pb2.LReLU:
    return lrelu
  elif op_type == types_pb2.ELU:
    return elu
  elif op_type == types_pb2.SELU:
    return selu
  elif op_type == types_pb2.Tanh:
    return tanh
  elif op_type == types_pb2.HardTanh:
    return hard_tanh
  elif op_type == types_pb2.Sigmoid:
    return sigmoid
  elif op_type == types_pb2.Softmax:
    return softmax
  else:
    raise ValueError("The given OpType %s is not an activation function." %
                     OpType.Name(op_type))

def relu(input_tensor, name="relu"):
  return common.add_node(
      name=name, op=types_pb2.ReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def lrelu(input_tensor, slope=0.2, name="lrelu"):
  params = node_pb2.Params()
  params.act_params.lrelu_params.slope = slope
  return common.add_node(
      name=name, op=types_pb2.LReLU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def elu(input_tensor, alpha=0.1, name="relu"):
  params = node_pb2.Params()
  params.act_params.elu_params.alpha = alpha
  return common.add_node(
      name=name, op=types_pb2.ELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def selu(input_tensor, alpha=1.6733, lambda_param=1.0507, name="selu"):
  params = node_pb2.Params()
  params.act_params.elu_params.alpha = alpha
  params.act_params.elu_params.lambda_param = lambda_param
  return common.add_node(
      name=name, op=types_pb2.SELU, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def tanh(input_tensor, name="tanh"):
  return common.add_node(
      name=name, op=types_pb2.Tanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def hard_tanh(input_tensor, min=-1, max=1, name="hard_tanh"):
  params = node_pb2.Params()
  params.act_params.hard_tanh_params.min = min
  params.act_params.hard_tanh_params.max = max
  return common.add_node(
      name=name, op=types_pb2.HardTanh, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout, params=params)[0]

def sigmoid(input_tensor, name="sigmoid"):
  return common.add_node(
      name=name, op=types_pb2.Sigmoid, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]

def softmax(input_tensor, name=None):
  input_tensor = common.check_and_add_layout_transform(
      name=name, op=types_pb2.Softmax, input_tensors=[input_tensor])[0]
  return common.add_node(
      name=name, op=types_pb2.Softmax, input_tensors=[input_tensor],
      output_tensors_dims=[input_tensor.shape.dims],
      output_tensor_layout=input_tensor.shape.layout)[0]
