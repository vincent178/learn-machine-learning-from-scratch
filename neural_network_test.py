import numpy as np
from neural_network import mean_squared_error, cross_entropy_error

def test_mean_square_error():
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert mean_squared_error(y, t) == 0.09750000000000003

def test_cross_entropy_error():
    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert cross_entropy_error(y, t) == 0.510825457099338
