# -*- encoding:utf8 -*-

"""
手写数字识别
使用学习到的参数，实现神经网络的推理过程，这个推理过程称为神经网络的前向传播
"""

import os, sys
from dataset.mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
from utils.functions import softmax, sigmoid

sys.path.append(os.pardir)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_date():
    """
    获取数据
    :return:
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)

    return x_test, t_test


def init_network():
    """
    权重，学习到的参数
    sample_weight2.pkl  python2.7 读取；sample_weight Python3读取
    :return:
    """
    with open("sample_weight2.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    """
    各个标签对应的概率
    :return:
    """
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


def acc_show():
    """
    概率精确度
    :return:
    """
    # 测试数据
    x, t = get_date()
    network = init_network()

    # 推送正确的个数
    acc_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        # 概率最高的元素索引
        p = np.argmax(y)
        if p == t[i]:
            acc_cnt += 1

    print 'acc:', str(float(acc_cnt) / len(x))


def acc_show_batch():
    """
    批处理
    :return:
    """
    x, t = get_date()
    network = init_network()

    # 推送正确的个数
    acc_cnt = 0
    batch_size = 100

    for i in range(0, len(x), batch_size):
        x_batch = x[i: i+ batch_size]
        y_batch = predict(network, x_batch)

        p = np.argmax(y_batch, axis=1)

        acc_cnt += np.sum(p == t[i:i+batch_size])

    print 'acc:', str(float(acc_cnt) / len(x))


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    print x_train.shape

    img = x_train[1000]
    label = t_train[1000]

    print label
    print img.shape

    img = img.reshape(28, 28)
    print img.shape

    img_show(img)

    acc_show()
    acc_show_batch()
