import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #cpu연산을 더빠르게 처리 하기 위해 입력해준다.

data = np.random.random((1000, 32))
label = np.random.random((1000, 10))

class MyModel(tf.keras.Model):
    def __init__(self, num_class = 10):#출력할 클래스 갯수를 입력 받는다.(미입력시 10이 자동으로 들어간다.)
        super(MyModel,self).__init__(name = 'my_model1')#식별할 이름 명을 지정
        self.num_class = num_class
        #층 설정
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')#레이어층 설정, 활성화 함수 설정
        self.dense_2 = tf.keras.layers.Dense(self.num_class, activation='sigmoid')#출력층
#노트: 정방향 패스를 항상 명령형 프로그래밍 방식으로 실행하려면 super 객체의 생성자를 호출할 때 dynamic=True를 지정하세요.
    def call(self,inputs):
        #정방향 패스를 정의
        #__init__에서 정의한 층을 사용
        x = self.dense_1(inputs)
        return self.dense_2(x)



#모델 클래스 객체 생성
model = MyModel(num_class=10)

#컴파일 단계는 훈련 과정을 설정하는 것이다(경사 하강법 정의)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
loss='categorical_crossentropy',metrics=['accuracy'])

#훈련 단계
model.fit(data,label, batch_size=32, epochs=5)


