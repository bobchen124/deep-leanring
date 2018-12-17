# -*- encoding:utf8 -*-

"""
简单层的实现 -- 购买苹果、梨
"""


from layer_naive import MulLayer, AddLayer


apple = 100
apple_num = 2

orange = 150
orange_num = 3

tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

print price

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
print dall_price, dtax
dapple_price, doragne_price = add_apple_orange_layer.backward(dall_price)
print dapple_price, doragne_price

dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print dapple, dapple_num

dorange, dorange_num = mul_orange_layer.backward(doragne_price)
print dorange, dorange_num