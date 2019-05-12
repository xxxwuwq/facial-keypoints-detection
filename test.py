import matplotlib.pyplot as plt
from models import CNN_model
from datasets import test_dataset


def test():
    # 加载网络结构
    model = CNN_model((96, 96, 1))
    # 加载模型权重
    model.load_weights('./ckpts/best_model.h5')
    # 加载测试数据集
    test = test_dataset()
    num = test['test'].shape[0]
    data = test['test']

    for i in range(num):
        # 训练时对图像进行[0， 1]归一化；对关键点进行（point - 48) / 48的归一化
        # 因此进行预测后，需要将预测的关键点坐标还原回来
        prediction = model.predict(data[i].reshape(1, 96, 96, 1) / 255.0 )[0]
        prediction = prediction * 48.0 + 48.0
        # 显示关键点预测结构
        plt.imshow(data[i].reshape(96, 96), cmap='gray')
        plt.scatter(prediction[0:30:2], prediction[1:30:2], c='r')
        plt.show()


if __name__ == '__main__':
    test()