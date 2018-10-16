# -*- encoding:utf8 -*-

"""
多维数组运算
"""

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print A.shape, B.shape

print A * B, (A * B).shape

# 乘积， 点积
print np.dot(A, B)

A1 = np.array([[1, 2], [3, 4], [5, 6]])
B1 = np.array([[5, 6, 7, 8], [7, 8, 5, 6]])

# (3, 2) (2, 4)
print A1.shape, B1.shape

# (3, 4)
C = np.dot(A1, B1)
print C, np.shape(C)

"""
神经网络的内积，要注意X，W，Y 的形状，特别是 X、W 的对应的元素个数是否一致，这点很重要
"""
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

print 'X shanpe = ', X.shape, ';W shape = ', W.shape

Y = np.dot(X, W)
print 'Y shape = ', Y.shape
print Y
