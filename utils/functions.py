# -*- encoding:utf8 -*-

"""
公用函数
"""

import numpy as np


def sigmoid(x):
    """
    sigmoid 函数实现
    :param x:
    :return:
    """
    return 1/(1 + np.exp(-x))


def rule(x):
    """
    rule函数
    :param x:
    :return: x < 0 -->0 ;  x > 0 -- > x
    """
    # maximum 输出较大值
    return np.maximum(0, x)


def identity_fun(val):
    """
    恒等函数
    :param val:
    :return:
    """
    return val


def softmax(a):
    """
    softmax 函数优化，防止数据溢出
    :param a:
    :return: 每一个值得
    """
    c = np.max(a)
    # 溢出对策
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    """
    交叉误差, 监督数据非标签形式，非 one_hot 表示
    :param y: 神经网络的输出
    :param t: 监督数据
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # one-hot-vector
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
