import numpy as np
from pandas.io.parsers import read_csv
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def train_dataset():
    """
    加载训练数据集
    :return: dict(train=train, val=val)
    """
    train_csv = './data/training.csv'
    dataframe = read_csv(os.path.expanduser(train_csv))
    dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    dataframe = dataframe.dropna()  # 将缺失值丢弃掉

    data = np.vstack(dataframe['Image'].values)  # 归一化数据,将array类型转化为list类型
    label = dataframe[dataframe.columns[:-1]].values  # 从第一列到倒数第二列，输出五官当中的label区域位置,.values这个功能是将DataFrame转化为np.array
    # 随机shuffle
    data, label = shuffle(data, label, random_state=0)

    num = data.shape[0]
    # 将训练数据集划8：2分为训练集和验证集
    train = dict(images=data[0:int(num * 0.8)], labels=label[0:int(num * 0.8), :])
    val = dict(images=data[int(num * 0.8):], labels=label[int(num * 0.8):, :])

    return dict(train=train, val=val)


def test_dataset():
    """
    加载测试数据集
    :return: dict(test=test)
    """
    test_csv = './data/test.csv'
    dataframe = read_csv(os.path.expanduser(test_csv))
    dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    dataframe = dataframe.dropna()

    return dict(test=np.vstack(dataframe['Image'].values))


def train_generator(images, labels, batch_size):
    """
    训练数据生成器
    :param images: numpy.ndarray, 图像数据, (n, 96 * 96)
    :param labels: numpy.ndarray, 关键点标签, (n, 30)
    :param batch_size: 批数量
    :return: 生成器
    """
    while True:
        count = 0
        x, y = [], []
        for i in range(images.shape[0]):
            img = images[i] / 255.0
            lable = (labels[i] - 48.0) / 48.0
            x.append(img)
            y.append(lable)
            count += 1

            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 96, 96, 1).astype("float32")
                y = np.array(y)
                yield x, y
                x, y = [], []


if __name__ == '__main__':
    data = train_dataset()
    train_datas, train_labels = data['train']['images'], data['train']['labels']
    val_datas, val_labels = data['val']['images'], data['val']['labels']
    print(train_datas.shape, train_labels.shape)
    for i in range(2):
        plt.imshow(train_datas[i].reshape((96, 96)))
        plt.scatter(train_labels[i, 0:30:2], train_labels[i, 1:30:2], c='r')
        plt.show()
