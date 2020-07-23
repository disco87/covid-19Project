import tensorflow as tf
import numpy as np
from tensorflow import keras

data = np.random.random((1000, 32))
label = np.random.random((1000, 10))
#############맞춤형 층 만들기 ##################
#1. tf.keras.Layer 를 상속 받는다
#2. __init__ : 이 층에서 사용되는 하위 층을 정의
#3. build : 층의 가중치를 만든다. add_weight 메서드를 이용하여 가중치를 추가
#3. call : 정방향 pass 를 구현
####아래의 코드는 입력과 커널 행렬의 matual(행열곱) 계산을 구현한 맞춤형 층의 예
class MyLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim 
        print('self.dim data:',self.output_dim)
        print('**kwargs:', kwargs)
        super(MyLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        #이층에서 훈련할 가중치 변수를 생성(add_weight 메소드를 사용)
        self.kernel = self.add_weight(name='kernel',shape=([input_shape[1]],self.output_dim),
        initializer='uniform',trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.kernel)#입력값 * w(가중치)
    def get_config(self):
        base_config = super(MyLayer,self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

################맞춤형 층을 사용하여 모델을 만든다#########################
#1. model 생성(신경망 생성)
#2. 컴파일 (경사하강법적용) - 훈련과정을 설정
#3. 훈련 

#신경망층 생성
model = tf.keras.Sequential([MyLayer(10), tf.keras.layers.Activation('relu')])#첫번째 인자는 output_dim, 사용할 활성 함수 지정
#훈련과정 설정 - 경사 하강법적용
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#학습 진행
model.fit(data,label,batch_size=32, epochs=5)



####################################################
#############콜백 함수 사용 하기######################
# - 훈련하는 동안 모델의 동작을 변경하고 확장하기위해 전달하는 객체 
# - 직접 만들거나 tf.keras.callbacks을 사용 할 수 있음
#  사용 가능한 메소드
# tf.keras.callbacks.ModelCheckpoint: 일정 간격으로 모델의 체크포인트를 저장합니다.
# tf.keras.callbacks.LearningRateScheduler: 학습률(learning rate)을 동적으로 변경합니다.
# tf.keras.callbacks.EarlyStopping: 검증 성능이 향상되지 않으면 훈련을 중지합니다.
# tf.keras.callbacks.TensorBoard: 텐서보드를 사용하여 모델을 모니터링합니다.
#
# 사용 시 fit메소드 안에서 지정 해준다.

callbacks = [
    # val_loss 가 2번의 에포크에 걸쳐 향상되지 않으면 훈련 중지
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
    # './logs' 디렉토리에 텐서보드 로그를 기록
    #tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data,label,batch_size=32,epochs=5,callbacks=callbacks)


#####################저장과 복원#######################
# 가중치를 텐서플로의 체크포인트 파일로 저장
model.save_weights('./weights/my_model')#가중지 w 저장
#모델의 상태를 봉원
#모델의 구조가 동일 해야 한다.
model.load_weights('./weights/my_model')

# 가중치를 HDF5 파일로 저장합니다.
model.save_weights('my_model.h5', save_format='h5')

# 모델의 상태를 복원합니다.
model.load_weights('my_model.h5')

# 모델을 JSON 포맷으로 직렬화합니다.
#YAML 포맷으로 직렬화하려면 텐서플로를 임포트하기 전에 pyyaml을 설치해야 합니다:(yaml 포맷으로 저장)


# 전체 모델을 HDF5 파일로 저장합니다.
model.save('my_model.h5')

# 가중치와 옵티마이저를 포함하여 정확히 같은 모델을 다시 만듭니다.
model = tf.keras.models.load_model('my_model.h5')

#########################################
###############다중 GPU 처리

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
############보통때와 같이 훈련을 하면 된다.
x = np.random.random((1024, 10))
y = np.random.randint(2, size=(1024, 1))
x = tf.cast(x, tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(buffer_size=1024).batch(32)

model.fit(dataset, epochs=1)