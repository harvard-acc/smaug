"""Smaug test framework for the Python client."""

import unittest

class SmaugTest(unittest.TestCase):
  """Smaug test base class.

  This class implements the common functions for different tests. Every test
  will inherit this class.
  """

  def get_node(self, graph, node_name):
    """ Find the node in the graph by its name."""
    for i in range(len(graph.nodes)):
      if graph.nodes[i].name == node_name:
        return graph.nodes[i]
