# -*- encoding:utf8 -*-

"""
激活函数
"""

import numpy as np
from utils.functions import *


class Relu:
    """
    激活函数 Relu ，
    表达式 ： y = x (x> 0) , y = 0 (x<=0)
    导数 ：dy/dx = 1（x>0） dy = 0 (x<=0)
    """
    def __init__(self):
        """
        mask True/False 组成的 numpy 数组
        """
        self.mask = None

    def forward(self, x):
        """
        :param x: numpy数组
        :return:
        """
        self.mask = (x <= 0)
        out = x.copy()
        # 小于0的变为0
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        :return:
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """
    y = 1 / (1 + exp(-x))
    :return:
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    """
    affine, 矩阵乘积运算
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss():
    """
    SoftmaxWithLoss
    """
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
