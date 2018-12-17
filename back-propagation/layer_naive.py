# -*- encoding:utf8 -*-

"""
简单层的实现 -
"""


class MulLayer:
    """
    乘法
    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x *y

        return out

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    """
    加法
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
