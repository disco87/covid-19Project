import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.utils import np_utils

mnist = tf.keras.datasets.mnist
# mnist dataset을 load한다.
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# float로 변환하고 minmax 스케일링을 한다. 이는 이미지 전처리의 가장 보편적인 방법 중 하나이다.
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
print(x_train.shape, x_train.dtype)
# y 값을 one-hot-encoding로 변환해준다.
y_unique_num = len(np.unique(y_train))
y_train = np_utils.to_categorical(y_train, y_unique_num)
y_test = np_utils.to_categorical(y_test, y_unique_num)
y_train[:5]

# test로 이미지를 한번 출력해보자.
r = random.randint(0, x_train.shape[0] - 1)
plt.imshow(
    x_train[r].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest" # 중간에 비어있는 값 처리
)
plt.show()