import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def neural_network():
    learnrate = 0.5
    x = np.array([1, 2])
    y = np.array(0.5)

    # Initial weights
    w = np.array([0.5, -0.5])

    activation_function = np.dot(w, x)
    # Calculate one gradient descent step for each weight 
    # TODO: Calculate output of neural network
    nn_output = sigmoid(activation_function)

    # TODO: Calculate error of neural network
    error = y - nn_output

    error_term = error * (nn_output * (1 - nn_output))

    # TODO: Calculate change in weights
    del_w = learnrate * error_term * x

    print('Neural Network predicted output:')
    print(nn_output)
    print('Amount of Error (actual_output - predicted_output):')
    print(error)
    print('Change in Weights:')
    print(del_w)


a = np.matrix('6,3,0 ; 2,5,1 ; 9,8,6')
b = np.matrix('7 ; 6 ; 5')

print(a * b)


def backward_propagation():
    def sigmoid(variable: np.array) -> np.array:
        return 1 / (1 + np.exp(-variable))

    x = np.array([0.5, 0.1, -0.2])
    target = 0.6 # actual output
    learnrate = 0.5

    weights_input_hidden = np.array([[0.5, -0.6],
                                     [0.1, -0.2],
                                     [0.1, 0.7]])

    weights_hidden_output = np.array([0.1, -0.3])

    # input activation function
    𝒽_input = np.dot(x, weights_input_hidden)
    output_hidden_layer = sigmoid(𝒽_input)
    # output activation function
    𝒽_output = np.dot(output_hidden_layer, weights_hidden_output)
    output = sigmoid(𝒽_output)

    error_output = target - output
    # gradient descent of the output
    ƍ_output = error_output * output * (1 - output)
    # gradient descent of the hidden layer
    ƍ_hidden = weights_hidden_output * ƍ_output * output_hidden_layer * (1 - output_hidden_layer)

    Δ_weight_output = learnrate * ƍ_output * output_hidden_layer
    Δ_weight_hidden = learnrate * ƍ_hidden * x[:, None]

    print("Δ_weight output: \n", Δ_weight_output)
    print("Δ_weight hidden: \n", Δ_weight_hidden)


backward_propagation()
