import tensorflow as tf
from tensorflow import keras
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

#함수형 API를 사용하여 머신러닝 만들기

inputs = tf.keras.Input(shape=(32,))#입력 플레이스홀더를 반환 - 32열으 1x32 행열생성

#######################################################
#다음 코드는 함수형 API를 사용하여 간단한 완전 연결 네트워크를 만드는 예입니다:
#층 객체는 텐서를 사용하여 호출되고 텐서를 반환합니다.
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
# x의 변수에 활성화 함수 relu를 이용해 inputs 데이터를 연수후 x에 삽입
x = tf.keras.layers.Dense(64,activation='relu')(x)
predictions = tf.keras.layers.Dense(10,activation='softmax')(x)

##############################################
#입력과 출력을 사용해 모델의 객체를 만드는 방법
model = tf.keras.Model(inputs = inputs, outputs = predictions)

#컴파일 단계 : 훈련과정을 설정해준다#경사 하강법
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#훈련 : epochs 의 설정된 횟수 만큼 반복
model.fit(data,labels,batch_size=32, epochs=5)







