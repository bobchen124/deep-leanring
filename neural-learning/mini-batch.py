# -*- encoding:utf8 -*-

"""
神经网络学习-损失函数
"""

import numpy as np
import matplotlib.pyplot as plt


def cross_entropy_error(y, t):
    """
    交叉误差
    :param y: 神经网络的输出
    :param t: 监督数据
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_not_one_hot(y, t):
    """
    交叉误差, 监督数据非标签形式，非 one_hot 表示
    :param y: 神经网络的输出
    :param t: 监督数据
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = t.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(f, x):
    """
    数值微分  导数
    :param f: 求导函数
    :param x: 数值
    :return:
    """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def fun_1(x):
    """
    y = 0.01x^2 + 0.1x
    :param x:
    :return:
    """
    return 0.01 * x ** 2 + 0.1 * x


def fun_2(x):
    """
    偏导数, f(x1, x2) = x1^2 + x2^2
    :param x: np 数组
    :return:
    """
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


def fun_tmp1(x):
    return x * x + 4.0 ** 2.0


def fun_tmp2(x):
    return 3.0 ** 2.0 + x * x


if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = fun_1(x)

    print numerical_diff(fun_1, 5)
    print numerical_diff(fun_1, 10)

    # x1 = 3, x2 = 4
    print numerical_diff(fun_tmp1, 3.0)
    print numerical_diff(fun_tmp2, 4.0)

    plt.xlabel("x")
    plt.ylabel("f(x)")

    plt.plot(x, y)
    plt.show()

