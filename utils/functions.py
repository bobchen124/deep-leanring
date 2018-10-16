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
