from keras.optimizers import *
from keras.callbacks import *
from models import CNN_model
from datasets import train_dataset, train_generator


def train():
    """
    训练程序
    :return:
    """
    # 网络结构加载
    model = CNN_model((96, 96, 1))
    # 优化器加载
    optimizer = SGD(lr=0.03, momentum=0.9, nesterov=True)
    # loss function
    model.compile(loss='mse', optimizer=optimizer)
    epoch_num = 1000
    learning_rate = np.linspace(0.03, 0.01, epoch_num)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    check_point = ModelCheckpoint('./ckpts/best_model.h5', monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='auto', period=1)
    # 训练数据, batch_size
    data = train_dataset()
    batch_size = 16
    # 训练集和验证集合的样本数量
    train_num = data['train']['images'].shape[0]
    val_num = data['val']['images'].shape[0]
    # 训练集和验证集生成器
    train_gen = train_generator(data['train']['images'], data['train']['labels'], batch_size)
    val_gen = train_generator(data['val']['images'], data['val']['labels'], batch_size)

    # 启动训练
    model.fit_generator(train_gen, steps_per_epoch=int(train_num / batch_size) + 1,
                    epochs=epoch_num, verbose=1, validation_data=val_gen,
                    validation_steps=int(val_num / batch_size) + 1, callbacks=[change_lr, early_stop, check_point])


if __name__ == '__main__':
    train()