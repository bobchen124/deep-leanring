# -*- encoding:utf8 -*-

"""
mnist数据
源码是 Python3.x 编写，现在是Python2.7
"""

import os.path
import gzip
import pickle
import os
import numpy as np
try:
    # import urllib.request   py3.x
    import urllib
except ImportError:
    # raise ImportError('You should use Python 3.x')
    raise ImportError('You should use Python 2.7')

# 下载地址
url_base = 'http://yann.lecun.com/exdb/mnist/'
# 下载文件名
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

# 文件目录
ds_dir = os.path.dirname(os.path.abspath(__file__))
save_file = ds_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    """
    文件下载，保存再当前目录
    :param file_name:
    :return:
    """
    file_path = ds_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print "Downloading " + file_name + " ... "
    # urllib.request.urlretrieve(url_base + file_name, file_path)
    urllib.urlretrieve(url_base + file_name, file_path)
    print "Done"


def download_mnist():
    """
    调用下载
    :return:
    """
    for v in key_file.values():
        _download(v)


def _load_label(file_name):
    """
    读取标签
    :param file_name:
    :return:
    """
    file_path = ds_dir + "/" + file_name

    print "Converting " + file_name + " to NumPy Array ..."
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print"Done"
    return labels


def _load_img(file_name):
    """
    读取图片
    :param file_name:
    :return:
    """
    file_path = ds_dir + "/" + file_name

    print "Converting " + file_name + " to NumPy Array ..."
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)

    print "_load_img Done"
    return data


def _convert_numpy():
    """

    :return:
    """
    dataset = {}

    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    """
    初始化
    :return:
    """
    download_mnist()

    dataset = _convert_numpy()

    print "Creating pickle file ..."
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

    print "init_mnist Done!"


def _change_one_hot_label(x):
    """
    更改数据
    :param x:
    :return:
    """
    t = np.zeros((x.size, 10))

    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    MNIST数据，（训练图像，训练标签），（测试图像，测试标签）
    Parameters
    ----------
    normalize : 是否将输入图像正规化为0.0~1.0的值
    one_hot_label :
        one_hot_label 是否奖标签保存为 one-hot 表示
        one-hot 表示仅正确的标签为1、像[0,0,1,0,0,0,0,0,0,0]这样
    flatten : 是否展开输入图像（变为一维数组）
    Returns
    -------
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
