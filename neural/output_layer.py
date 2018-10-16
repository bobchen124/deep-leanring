# -*- encoding:utf8 -*-

"""
输出层的设计
神经网络可以分为分类问题和回归问题，需要根据情况改变输出层的激活涵涵
一般而言，回归问题用恒等函数，分类问题用 softmax 函数，softmax 输出解释为概率
softmax = exp(a[k]) / sum(exp(a[0..i]))
sum(exp(a[0..i])) :  求和
"""

import numpy as np


def softmax_fun(a):
    """
    softmax 函数
    :param a:
    :return: 每一个值得
    """
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def softmax_fun_op(a):
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


if __name__ == '__main__':
    an = np.array([0.3, 2.9, 4.0])
    print softmax_fun(an)

    ar = np.array([1010, 1000, 990])
    # 会发生溢出
    print softmax_fun(ar)
    print softmax_fun_op(ar)
