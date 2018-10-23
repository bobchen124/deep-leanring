# -*- encoding:utf8 -*-

"""
神经网络梯度
"""

import os
import sys
import numpy as np

from utils.functions import softmax, cross_entropy_error
from utils.gradient import numerical_gradient

sys.path.append(os.pardir)


class SimpleNet:
    """
    简单神经网络梯度
    """
    def __init__(self):
        # 权重参数
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """
        用于预测
        :param x: np数组
        :return:
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        计算损失函数
        :param x:
        :param t: 正确标签
        :return:
        """
        z = self.predict(x)
        y = softmax(z)

        loss = cross_entropy_error(y, t)
        return loss


if __name__ == '__main__':
    net = SimpleNet()
    print net.W

    x = np.array([0.6, 0.9])
    p = net.predict(x)

    print p
    print np.argmax(p)

    t = np.array([0, 0, 1])
    x = np.array([0.6, 0.9])
    print net.loss(x, t)
