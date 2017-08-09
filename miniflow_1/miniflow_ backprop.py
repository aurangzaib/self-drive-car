import numpy as np


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        # for each node of Input class
        # set the value
        # for other classes, the value is set in forward_pass function
        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(output_node, sorted_nodes):
    for node in sorted_nodes:
        node.forward()
    for node in sorted_nodes[::-1]:
        node.backward()


class Node(object):
    """
    Node class provides the base set of properties
    we need to define subclasses for calculations
    """

    def __init__(self, inbound_nodes=[]):
        # nodes from which _this_ node receives values (input)
        self.inbound_nodes = inbound_nodes
        # nodes to which _this_ node passes values
        self.outbound_nodes = []
        for n in self.inbound_nodes:
            # each inbound node have outbound nodes
            n.outbound_nodes.append(self)
        # each node eventually calculates some value
        # setting initially as None
        self.value = None
        self.gradients = {}

    # each node performs forward propagation
    def forward(self):
        raise NotImplemented

    # each node performs backward propagation
    def backward(self):
        raise NotImplemented
