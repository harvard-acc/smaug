from smaug.python.ops import *

class AttentionBase:
  def __init__(self, memory, w_encoder, w_decoder, name="attention"):
    """Construct base Attention class.

    Args:
      name: Name to use when creating ops.
      memory: The memory to query; usually the outputs of an RNN encoder. The
        tensor should be shaped [batch, time, depth].
      w_encoder: The weight used for the memory layer shaped [depth, depth].
      w_decoder: The weight used for the query layer shaped [depth, depth].
    """
    self.name = name + ":"
    self.batch_size = memory.shape.dims[0]
    self.timesteps = memory.shape.dims[1]
    self.depth = memory.shape.dims[2]
    self.w_encoder = w_encoder
    self.w_decoder = w_decoder
    self.memory = memory
    self.keys = self._memory_layer(memory)

  def _memory_layer(self, memory):
    name_pfx = self.name + ":mem_layer:"
    # Reshape memory from [batch, time, depth] to [batch * time, depth].
    memory = reshape(
        memory, [self.batch_size * self.timesteps, self.depth], NC,
        name=name_pfx + "reshape0")
    # scores shaped [batch * time, depth].
    scores = mat_mul(memory, self.w_encoder, name=name_pfx + "mm")
    # [batch * time, depth] -> [batch, time, depth]
    return reshape(
        scores, [self.batch_size, self.timesteps, self.depth], NTC,
        name=name_pfx + "reshape1")

  def _query_layer(self, query):
    # Return a tensor shaped [batch, depth].
    return mat_mul(query, self.w_decoder, name=self.name + "query_layer")

  def __call__(self, query):
    """ Invoke the attention layer to compute the attention vector."""

    # Compute alignments shaped [batch, time].
    alignment = self._compute_alignment(query)

    # Compute context vector (aka attention). Context is the inner product of
    # alignments and keys along the time dimension. The shape of context is
    # [batch, depth].
    # alignment_batches is shaped [1, time] * batch.
    alignment_batches = split(
        alignment, self.batch_size, axis=0, name=self.name + "split")
    # [batch, time, depth] -> [batch, depth, time] -> [depth, time] * batch.
    values = unstack(reorder(self.memory, NCT), 0, name=self.name + "unstack")
    context = []
    for i in range(self.batch_size):
      # Every mat_mul produces a tensor shaped [1, depth].
      context.append(
          mat_mul(alignment_batches[i], values[i], name=self.name + "mm"))
    # context shaped [batch, depth].
    context = concat(context, 0, name=self.name + "concat")

    return context

  def _compute_alignment(self, query):
    query = self._query_layer(query)
    score = self.compute_score(query)
    alignment = softmax(score, name=self.name + "softmax")
    return alignment

  def compute_score(self, query):
    raise NotImplementedError(
        "This class should be overridden by child classes.")

class BahdanauAttention(AttentionBase):
  """Implements Bahdanau attention.

  The attention implementation is described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  """
  def __init__(self,
               memory,
               w_encoder,
               w_decoder,
               w_alignment,
               name="bahdanau_attention"):
    AttentionBase.__init__(self, memory, w_encoder, w_decoder, name)
    # Alignment weight shaped [1, depth].
    self.w_alignment = w_alignment

  def compute_score(self, query):
    # The score is computed as tanh(query + keys) * w_alignments.

    name_pfx = self.name + ":score:"
    # Reshape from [batch, depth] to [batch, 1, depth] for broadcasting.
    query = expand_dims(query, 1, name=name_pfx + "expand")
    # [batch, time, depth].
    activations = tanh(
        add(self.keys, query, name=name_pfx + "add"), name=name_pfx + "stack")
    # [batch * time, depth]
    activations = reshape(
        activations, [self.batch_size * self.timesteps, self.depth], NC,
        name=name_pfx + "reshape0")
    # [batch * time, 1]
    scores = mat_mul(activations, self.w_alignment, name=name_pfx + "mm")
    # [batch, time]
    scores = reshape(
        scores, [self.batch_size, self.timesteps],
        NC,
        name=name_pfx + "reshape1")
    return scores
