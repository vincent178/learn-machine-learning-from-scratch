import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.rand(5, 3)
print(x)

x = torch.tensor([1, 2, 3, 4])
print(x)
print(x.shape)


def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

x = np.arange(0, 3, 0.1)    
print(x)

def step_function(x):
    y = x > 0
    return y.astype(int)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)

    plt.plot(x, y1, linestyle='--', label='Step')
    plt.plot(x, y2, color='blue', label='Sigmoid')
    plt.plot(x, y3, label='ReLU')
    plt.xlabel('Activation Function')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


A = np.array([1, 2, 3, 4])
A = A.reshape(2, 2)
print(A)

B = np.array([[5, 6], [7, 8]])
print(B)

C = np.dot(A, B)
print(C)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
x = np.array([0.1, 0.2])
y = np.array([[0.1, 0.2]])
z = np.array([0.3, 0.4])
print(x.shape)
print("W1 shape: ", W1.shape)
a1 = np.dot(x, W1)
print(a1)
print("a1 shape: ", a1.shape)

a2 = np.dot(y, W1)
print(a2.shape)
print(a2)

a3 = np.dot(x, z)
print(a3.shape)
print(a3)

a4 = x * z
print(a4)

# print(np.dot(np.array([[1, 2]]), np.array([[3, 4]])))
