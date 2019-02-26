"""Here we define global variables.

Currently it contains:
  1) A global active graph
  2) Alignment information for various backends.
"""

# This keeps track of the current active graph. Currently we only support
# one active graph, we can change that if later we need multiple graphs.
active_graph = None

# Alignment information for various backends.
backend_alignment = {"Reference": 0, "SMV": 8}

def get_graph():
  """Obtain the current active graph."""
  return active_graph

def set_graph(graph):
  """Set the active graph."""
  global active_graph
  active_graph = graph

def clear_graph():
  """Clear the active graph.

  This will be used when the graph context is cleaned up.
  """
  global active_graph
  active_graph = None
