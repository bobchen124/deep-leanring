# -*- encoding:utf8 -*-

"""
感知机, 接收多个信号，输出一个信号
与门、与非门、或门
局限性：只能用一条直线分割空间
"""

import numpy as np


def and_fun(x1, x2):
    """
    简单实现
    :param x1: 1 or 1
    :param x2: 1 or 1
    :return: 0 or 1
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2

    if tmp > theta:
        return 1

    return 0


def and2_fun(x1, x2):
    """
    使用权重和偏置, sum(w * x) + b
    :param x1:
    :param x2:
    :return:
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    # 相乘求和
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0

    return 1


def nand_fun(x1, x2):
    """
    与非门
    :param x1:
    :param x2:
    :return:
    """
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    # 相乘求和
    tmp = np.sum(w * x) + b

    return 0 if tmp <= 0 else 1


def or_fun(x1, x2):
    """
    或门
    :param x1:
    :param x2:
    :return:
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    # 相乘求和
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


def xor_fun(x1, x2):
    """
    异或门，组合感知机实现异或门
    :param x1:
    :param x2:
    :return:
    """
    s1 = nand_fun(x1, x2)
    s2 = or_fun(x1, x2)

    return and2_fun(s1, s2)


print xor_fun(0, 0), xor_fun(0, 1), xor_fun(1, 0), xor_fun(1, 1)
