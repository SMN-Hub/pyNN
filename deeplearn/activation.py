import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """

    A = 1 / (1 + np.exp(-Z))

    return A


def tanh(Z):
    """
    Implement the hyperbolic tangent function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.tanh(Z)

    assert (A.shape == Z.shape)

    return A


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    return A


def leakrelu(Z):
    """
    Implement the Leaky Rectified Linear Unit function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """

    A = np.maximum(0.01 * Z, Z)

    assert (A.shape == Z.shape)

    return A


# Activation functions
ACTIVATIONS = {'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu, 'leakrelu': leakrelu}


def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- linear layer for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def tanh_backward(dA, Z):
    """
    Implement the backward propagation for a single TANH unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- linear layer for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    s = 1 - tanh(Z)**2
    dZ = dA * s

    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- linear layer for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def leakrelu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    Z -- linear layer for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    # When z <= 0, dz = 0.01
    derivative = np.ones(Z.shape)
    derivative[Z < 0] = 0.01

    dZ = dA * derivative

    assert (dZ.shape == Z.shape)

    return dZ


# Activation functions
BACKWARD_DERIVATIONS = {'sigmoid': sigmoid_backward, 'tanh': tanh_backward, 'relu': relu_backward, 'leakrelu': leakrelu_backward}

