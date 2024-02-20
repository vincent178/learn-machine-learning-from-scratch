import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.timer import Timer

def test():
    return (1, 2), (3, 4)

(a, b), (c, d) = test()

# v1 = np.array([1, 2, 3])
# print(v1.shape)
# print(v1.T.shape)


# a = np.array([1, 2])
# b = np.array([[3, 4],
#               [5, 6],
#               [7, 8]])

# print(a.dot(b))
# print(a.dot(b.T))

# print(np.dot(b, a))
# print(np.dot(b, a.T))
# print(b.dot(a))
# print(b.dot(a.T))
# print(np.dot(a, b.T))
# print(np.dot(a, b))

# v2 = np.array([4, 5, 6])
# print(v2.shape)

# print(np.dot(v1, v2))

# print(np.dot(v1.T, v2))

# v3 = np.array([[1, 2, 3]])
# print(np.dot(v3, v2))
# print(np.dot(v3.T, v2))


n = 10000
a = torch.ones([n])
b = torch.ones([n])

c = torch.ones(n)
timer = Timer()
for i in range(n): # 循环太慢了
    c[i] = a[i] + b[i]
print(f"{timer.stop():.5f} sec") # 0.02544 sec

timer.start()
d = a + b 
print(f"{timer.stop():.5f} sec") # 0.00001 sec



