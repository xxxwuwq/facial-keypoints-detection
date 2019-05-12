from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model

# 定义网络结构
def CNN_model(size):
    input_data = Input(size)
    x = Conv2D(32, (3, 3), activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    output = Dense(30)(x)
    return Model(inputs=input_data, outputs=output, name='Discriminator')


