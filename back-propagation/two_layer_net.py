# coding: utf-8

import sys, os
import numpy as np
from utils.layers import *
from utils.gradient import numerical_gradient
from collections import OrderedDict

sys.path.append(os.pardir)


class TwoLayerNet:
    """
    二层神经网络
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        初始化
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param weight_init_std:
        """
        # 初始化权重
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        进行识别推理，x 是图像数据
        :param x:
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        计算损失函数的值
        :param x: 输入数据
        :param t: 监督数据
        :return:
        """
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        计算识别精度
        :param x:
        :param t:
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        通过数值微分计算关于权重参数的梯度
        :param x:
        :param t:
        :return:
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """
        通过误差反向传播法计算关于权重参数的梯度
        :param x:
        :param t:
        :return:
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
