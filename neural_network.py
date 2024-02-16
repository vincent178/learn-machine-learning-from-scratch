import numpy as np
from utils import mnist
from matplotlib import pyplot as plt

# 经典统计学习技术中的线性回归和softmax回归可以视为线性神经网络
# https://zh.d2l.ai/chapter_linear-networks/index.html
(x_train, t_train), (x_test, t_test) = mnist.load_mnist()
print(x_train.shape)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    # np.log(0) = -inf, to prevent this, add a small delta 1e-7 (0.0000001)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def show_image(x):
    img = x.reshape(28, 28)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()

