import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from typing import Callable

# -- 3D

# x = np.linspace(-40, 40, 100)
# y = np.linspace(-40, 40, 100)
# x, y = np.meshgrid(x, y)
# z = 0.5 * x**2 + y**2

# # 创建3D图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis')

# # 显示图形
# plt.title("0.5x^2 + y^2")
# plt.show()

# exit()

# -- End 3D

# -- 3D with Animation

# a = np.linspace(0, 2 * np.pi)
# b = np.linspace(0, np.pi)
# x = 10 * np.outer(np.cos(a), np.sin(b))
# y = 10 * np.outer(np.sin(a), np.sin(b))
# z = 10 * np.outer(np.ones(np.size(a)), np.cos(b))

# # 创建动画
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# def update(frame):
#     ax.clear()
#     print(frame)
#     ax.plot_surface(x + frame, y, z, color='b')
#     ax.set_xlim(-8, 8)
#     ax.set_ylim(-8, 8)
#     ax.set_zlim(-8, 8)

# ani = animation.FuncAnimation(fig, update, frames=26, interval=100)
# plt.show()

# -- END 3D with Animation

def gradient_descent(start: float, gradient: Callable[[float], float],
                     learn_rate: float, max_iter: int, tol: float = 0.01):
    x = start
    steps = [start]  # history tracking

    for _ in range(max_iter):
        diff = learn_rate*gradient(x)
        if np.abs(diff) < tol:
            break
        x = x - diff
        steps.append(x)  # history tracing
  
    return steps, x

def func1(x):
    return np.power(x, 2) - 4 * x + 1

def gradient_func1(x):
    return 2 * x - 4

def func2(x):
    return np.power(x, 4) - 2 * np.power(x, 3) + 2

def gradient_func2(x):
    return 4 * np.power(x, 3) - 6 * np.power(x, 2)

def draw_plot_with_learning_rate(rate, func, gradient_func):
    fig = plt.figure()

    x = np.linspace(-0.5, 2.0, 100)
    y = func(x)

    plt.plot(x, y)

    steps, result = gradient_descent(-0.5, gradient_func, rate, 100)
    steps = np.array(steps)

    print(steps)
    print(result)

    # # 散点
    # plt.scatter(steps, func(steps), color='red')

    # # 连接散点
    # plt.plot(steps, func(steps), linestyle='-', color='red')

    # for i, _ in enumerate(steps):
    #     if i > 8:
    #         break
    #     # 标注
    #     plt.annotate(str(i), (steps[i], func(steps[i])), textcoords="offset points", xytext=(10, -7), ha='center')

    def update(frame):
        ani_x = steps[:frame]
        ani_y = func(ani_x)

        plt.scatter(ani_x, ani_y, color='red')
        plt.plot(ani_x, ani_y, linestyle='-', color='red')
        if frame > 8:
            return
        plt.annotate(str(frame), (steps[frame], func(steps[frame])), textcoords="offset points", xytext=(10, -7), ha='center')

    ani = animation.FuncAnimation(fig, update, frames=len(steps), interval=500)

    plt.title(f"Leanring rate {rate}")
    plt.grid()
    plt.show()

draw_plot_with_learning_rate(0.4, func2, gradient_func2)

