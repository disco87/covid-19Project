import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#pip install tensorflow==2.0.0-rc0
import cv2

path_basic = 'tensorflow/trImg/'
f_list_1 = os.listdir(path_basic + '1')
f_list_2 = os.listdir(path_basic + '2')
f_list_3 = os.listdir(path_basic + '3')
f_list_4 = os.listdir(path_basic + '4')
sum_img = []
test_img = [] 
y_train = []
y_test = []
shape_1 = 28

for i in f_list_1:
    img = cv2.imread(path_basic + '1/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    sum_img.append(img/255)
    y_train.append(1)
for i in f_list_2:
    print(y_train)
    img = cv2.imread(path_basic + '2/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    sum_img.append(img/255)
    y_train.append(2)
for i in f_list_3:
    img = cv2.imread(path_basic + '3/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    test_img.append(img/255)
    y_test.append(1)
for i in f_list_4:
    img = cv2.imread(path_basic + '4/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    test_img.append(img/255)
    y_test.append(2)

x_train = np.array(sum_img)
train_imgs = x_train.reshape((len(x_train),shape_1,shape_1,1))
y_train = np.array(y_train)
y_train = y_train.reshape(-1,1)
x_test = np.array(test_img)
test_imgs = x_test.reshape((len(x_test),shape_1,shape_1,1))
y_test = np.array(y_test)

train_ds = tf.data.Dataset.from_tensor_slices((train_imgs,y_train)).shuffle(len(train_imgs))

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1),padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D((2,2),padding='same')#2x2에서 가장 큰값만 나온다(압축)
        self.conv2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same')
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.pool2 = tf.keras.layers.MaxPooling2D((2,2),padding = 'same')
        self.flatten = tf.keras.layers.Flatten()
        self.dens1 = tf.keras.layers.Dense(64, activation='relu')
        self.dens2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.drop1(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dens1(x)
        return self.dens2(x)


model = MyModel()#모델 지정
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()#손실함수 지정
optimizer = tf.keras.optimizers.Adam()#경사 하강법 지정(ADAM)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape: #GradientTape 를 이용해 학습방법
        predictions = model(images)#Model에 이미지를 넣은후 넣은 결과 값을 predictions변수에 넣는다.
        loss = loss_object(labels, images)#손실 함수에 label과 image정보를 넣은후 손실을 t_loss변수에 넣는다

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))