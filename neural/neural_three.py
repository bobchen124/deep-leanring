# -*- encoding:utf8 -*-

"""
3层神经网络的实现
A = XW + B, 用多维数组来实现，输入信号、权重、偏置设置为任意值
"""

import numpy as np
from active_fun import sigmoid_fun


# 输入信息
x = np.array([1.0, 0.5])


def layer_one():
    """
    输入层到第一层的信号传递，  A1 运算
    :return:
    """
    # 权重
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # 偏置
    b1 = np.array([0.1, 0.2, 0.3])

    print w1.shape, b1.shape

    # 内积
    a1 = np.dot(x, w1) + b1
    print a1, a1.shape

    # 隐藏层被激活函数转换后，用Z表示
    z1 = sigmoid_fun(a1)
    print z1, z1.shape
    return z1


def layer_two(z1):
    """
    第一层到第二层的信号传递，
    :return:
    """
    # 权重
    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    # 偏置
    b2 = np.array([0.1, 0.2])

    print w2.shape, b2.shape

    # 内积
    a2 = np.dot(z1, w2) + b2
    print a2, a2.shape

    # 隐藏层被激活函数转换后，用Z表示
    z2 = sigmoid_fun(a2)
    print z2, z2.shape
    return z2


def identity_fun(val):
    """
    恒等函数
    :param val:
    :return:
    """
    return val


def layer_three(z2):
    """
    第二层到第三层的信号传递，
    :return:
    """
    # 权重
    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    # 偏置
    b3 = np.array([0.1, 0.2])

    # 内积
    a3 = np.dot(z2, w3) + b3
    print a3, a3.shape

    y = identity_fun(a3)
    print 'layer_three Y = ', y


def init_network():
    """
    代码整理
    :return:
    """
    network = {}

    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forword(network, x):
    """
    输入信号转换为输出信号
    :return:
    """
    a1 = np.dot(x, network['w1']) + network['b1']
    z1 = sigmoid_fun(a1)

    a2 = np.dot(z1, network['w2']) + network['b2']
    z2 = sigmoid_fun(a2)

    a3 = np.dot(z2, network['w3']) + network['b3']
    y = identity_fun(a3)

    print 'forword Y = ', y


if __name__ == '__main__':
    layer_three(layer_two(layer_one()))

    forword(init_network(), x)
