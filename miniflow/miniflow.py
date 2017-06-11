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


def forward_pass(output_node, sorted_nodes):
    for node in sorted_nodes:
        # setting value of the node in forward
        node.forward()
    return output_node.value


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

    # each node performs forward propagation
    def forward(self):
        """
        computes output values using inbound_nodes values
        :return: value
        """
        raise NotImplemented


class Input(Node):
    """
    Input class doesnt perform any calculations
    it just holds the input features and model parameters (weights/bias)
    """

    def __init__(self):
        Node.__init__(self)

    # example: val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, *input_array):  # feature, target
        # unlike Input class, Add class has inbound values (x, y)
        Node.__init__(self, np.array(input_array))

    def forward(self):
        val = 0
        for node in self.inbound_nodes:
            val += node.value
        self.value = val


class Mul(Node):
    def __init__(self, *input_array):  # feature, target
        # unlike Input class, Add class has inbound values (x, y)
        Node.__init__(self, np.array(input_array))

    def forward(self):
        val = 1
        for node in self.inbound_nodes:
            val *= node.value
        self.value = val


class Linear(Node):
    """
    finds activation function
    h
    """

    def __init__(self, *input_array):  # feature, target
        # unlike Input class, Add class has inbound values (x, y)
        Node.__init__(self, np.array(input_array))

    def forward(self):
        _inputs_, _weights_, _bias_ = self.inbound_nodes
        # activation function --> ∑(𝜘*𝜔)+𝙗
        self.value = np.dot(_inputs_.value, _weights_.value) + _bias_.value


class Sigmoid(Node):
    """
    find sigmoid of activation functions
    f(h)
    """

    def __init__(self, *input_array):  # feature, target
        # unlike Input class, Add class has inbound values (x, y)
        Node.__init__(self, np.array(input_array))

    @staticmethod
    def _sigmoid(variable):
        return 1. / (1. + np.exp(-variable))

    def forward(self):
        _inputs_ = self.inbound_nodes[0].value
        # activation function --> ∑(𝜘*𝜔)+𝙗
        self.value = self._sigmoid(_inputs_)


class MSE(Node):
    """
    find error in the output
    𝝃 --> error
    𝜹 --> gradient
    """

    def __init__(self, *input_array):  # feature, target
        # unlike Input class, Add class has inbound values (x, y)
        Node.__init__(self, np.array(input_array))

    def forward(self):
        # reshape(row, col) where -1 means no change
        _output_ = self.inbound_nodes[0].value.reshape(-1, 1)  # 1 col vector
        _prediction_ = self.inbound_nodes[1].value.reshape(-1, 1)  # 1 col vector
        _targets_ = len(self.inbound_nodes[0].value)  # m --> # of targets
        self.value = np.mean(np.square(_output_ - _prediction_))


x, y = Input(), Input()
add = Add(x, y)  # variadic arguments
# for list, use *
# for dictionary, use **
feed_dict = {x: 10, y: 20}
sorted_nodes = topological_sort(feed_dict=feed_dict)
output = forward_pass(add, sorted_nodes)
print("add output:", output)

inputs, weights, bias = Input(), Input(), Input()
linear = Linear(inputs, weights, bias)
sigmoid = Sigmoid(inputs, weights, bias)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])
feed_dict_linear = {
    inputs: X_, weights: W_, bias: b_
}

sorted_nodes_linear = topological_sort(feed_dict=feed_dict_linear)
output_linear = forward_pass(linear, sorted_nodes_linear)
output_sigmoid = forward_pass(sigmoid, sorted_nodes_linear)
print("linear combination:\n", output_linear)
print("sigmoid output:\n", output_sigmoid)

# using MSE to find error or cost
output, prediction = Input(), Input()
cost = MSE(output, prediction)
y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])
feed_dict = {output: y_, prediction: a_}
sorted_nodes_mse = topological_sort(feed_dict=feed_dict)
forward_pass(cost, sorted_nodes_mse)
print("MSE: ", cost.value)


def my_func(**arg):
    # to access each property
    print(arg)


a = {
    "hello": 1,
    "world: ": 2
}

# my_func(**a)
