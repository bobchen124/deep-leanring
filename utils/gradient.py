# -*- encoding:utf8 -*-

"""
深度学习---梯度
"""

import numpy as np


def fun_2(x):
    """
    偏导数, f(x1, x2) = x1^2 + x2^2
    :param x: np 数组
    :return:
    """
    # return x[0] ** 2 + x[1] ** 2
    return np.sum(x**2)


def numerical_gradient_1d(f, x):
    """
    梯度
    :param f: 输入函数
    :param x: numpy 数组
    :return:
    """
    # 0.0001
    h = 1e-4
    # 生成和 x 形状相同的数组
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)j计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原值
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降法
    :param f: 函数
    :param init_x: 初始值
    :param lr: 学习率
    :param step_num: 重复次数
    :return:
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


# print numerical_gradient(fun_2, np.array([3.0, 4.0]))
# print numerical_gradient(fun_2, np.array([0.0, 2.0]))
# print gradient_descent(fun_2, init_x=np.array([3.0, 4.0]), lr=0.1)
