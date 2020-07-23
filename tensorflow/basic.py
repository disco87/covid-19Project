import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train : 0 - 9 까지의 손글씨 이미지가 6만개 담긴다.(1개당 28x28)
#y_train : 0 - 9 까지의 손글씨 이미지에 대해 어떤숫자인지 0 - 1 까지의 레이블 정보가 담겨 있다.
#x_test : 0- 9 까지의 평가하게 될 이미지가 담긴다. 약 1만개 (1개당 28x28) 
#y_test : 0 - 9 까지의 손글씨 이미지에 대해 어떤숫자인지 0 - 1 까지의 레이블 정보가 담겨 있다.

x_train, x_test = x_train / 255.0, x_test / 255.0 # 흑백으로 만들기..


#신경망 층 생성
model = tf.keras.models.Sequential([


    tf.keras.layers.Flatten(input_shape=(28,28)), #입력 이미지가 28x28 이브로 1차원 텐서로 펼치기 위함
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),#2번째 층에서 걸러진 값중 20%를 버린다.
    tf.keras.layers.Dense(10, activation='softmax') 
    # 출력 값이며 층이 10개인 이유는 0-9중 의 10개의 클래스를 가져야 하기때문
])

    # '''
    #https://forensics.tistory.com/25 참조 
    # 총 4개의 레이어로 구성된 신경망인데, 1번째 레이어는 입력 이미지의 크기가 28×28이므로 이를 1차원 텐서로 
    # 펼치는 것이고, 2번째 레이어는 1번째 레이어에서 제공되는 784 개의 값(28×28)을 입력받아 128개의 값으로 
    # 인코딩해 주는데, 활성함수로 ReLU를 사용하도록 하였습니다. 2번째 레이어의 실제 연산은 1번째 레이어에서
    #  제공받은 784개의 값을 784×128 행렬과 곱하고 편향값을 더하여 얻은 128개의 출력값을 다시 ReLU 함수에 
    #  입력해 얻은 128개의 출력입니다. 3번째는 128개의 뉴런 중 무작위로 0.2가 의미하는 20%를 다음 레이어의 
    #  입력에서 무시합니다. 이렇게 20% 정도가 무시된 값이 4번째 레이어에 입력되어 충 10개의 값을 출력하는데, 
    #  여기서 사용되는 활성화 함수는 Softmax가 사용되었습니다. Softmax는 마지막 레이어의 결과값을 다중분류를 
    #  위한 확률값으로 해석할 수 있도록 하기 위함입니다. 10개의 값을 출력하는 이유는 입력 이미지가 0~9까지의 
    #  어떤 숫자를 의미하는지에 대한 각각의 확률을 얻고자 함입니다. 이렇게 정의된 모델을 학습하기에 앞서 
    #  다음처럼 컴파일합니다.
    # '''
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])#경사하강법
# >> (optimizer=훈련과정,loss=손실함수,metrics=훈련을 모니터링하기위해 사용)
# 참조 https://www.tensorflow.org/guide/keras/overview?hl=ko
# 모델의 학습 중에 역전파를 통한 가중치 최적화를 위한 기울기 방향에 대한 경사하강을 위한 방법으로 
# Adam을 사용했으며 손실함수로 다중 분류의 Cross Entropy Error인 ‘sparse_categorical_crossentropy’를 
# 지정하였습니다. 그리고 모델 평가를 위한 평가 지표로 ‘accuracy’를 지정하였습니다. 이제 다음처럼 모델을 
# 학습할 수 있습니다.
model.fit(x_train,y_train,epochs=7,validation_data=(x_test,y_test))
#(학습할 이미지,레이블, epochs=학습할 반복횟수, batch_size=샘플의 크기)
# 학습에 사용되는 데이터넷과 학습 반복수로 5 Epoch을 지정했습니다. Epoch은 전체 데이터셋에 대해서 


model.evaluate(x_test, y_test, verbose=2) #모델을 평가
#주어진 데이터로 추론 모드의 손실이나 지표를 평가합니다:

result = model.predict(x_test)
#주어진 데이터로 추론 모드에서 마지막 층의 출력을 예측하여 넘파이 배열로 반환합니다

print(result[0])


