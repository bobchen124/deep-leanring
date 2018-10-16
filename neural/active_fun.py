# -*- encoding:utf8 -*-

"""
激活函数
"""

import numpy as np
import matplotlib.pyplot as plt


def step_fun(x):
    """
    numpy 数组
    :param x: numpy 数组
    :return:
    """
    y = x > 0
    # print y
    return y.astype(np.int)


def step_function(x):
    """
    numpy 数组
    :param x: numpy 数组
    :return: 1 or 0
    """
    return np.array(x > 0, dtype=np.int)


def sigmoid_fun(x):
    """
    sigmoid 函数实现
    :param x:
    :return:
    """
    return 1/(1 + np.exp(-x))


def rule_fun(x):
    """
    rule函数
    :param x:
    :return: x < 0 -->0 ;  x > 0 -- > x
    """
    # maximum 输出较大值
    return np.maximum(0, x)


if __name__ == '__main__':
    # 阶跃函数的图形
    xr = np.arange(-5.0, 5.0, 0.1)
    ys = step_function(xr)
    y = sigmoid_fun(xr)

    plt.plot(xr, ys, linestyle="--")
    plt.plot(xr, y)

    plt.ylim(-0.1, 1.1)
    plt.show()
