import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
#pip install tensorflow==2.0.0-rc0
import cv2


# Func_Class.cam_init()



# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #mnist 데이터 셋 다운(테스트용)
# x_train = x_train.reshape((60000, 28,28, 1))#28x28 사이즈의 이미지로 만듬(60000개의 이미지가 있음), 1 : 흑백
# x_test = x_test.reshape((10000,28,28,1))
# train_imgs, test_imgs = x_train / 255, x_test/255 #각 이미지를 0 -1 사이의 값으로 바꾸기 위함



path_basic = 'tensorflow/trImg/'
f_list_0 = os.listdir(path_basic + '0')
f_list_1 = os.listdir(path_basic + '1')
f_list_2 = os.listdir(path_basic + '2')
f_list_3 = os.listdir(path_basic + '3')
f_list_4 = os.listdir(path_basic + '4')
sum_img = []
test_img = [] 
y_train = []
y_test = []
shape_1 = 28
# for i in f_list_0:
#     img = cv2.imread(path_basic + '0/' + i, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
#     sum_img.append(img/255)
#     y_train.append(0)
for i in f_list_1:
    img = cv2.imread(path_basic + '1/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    sum_img.append(img/255)
    y_train.append(1)
for i in f_list_3:
    img = cv2.imread(path_basic + '3/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    sum_img.append(img/255)
    y_train.append(3)
for i in f_list_2:
    print(y_train)
    img = cv2.imread(path_basic + '2/' + i, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(shape_1,shape_1),interpolation=cv2.INTER_LINEAR)
    sum_img.append(img/255)
    y_train.append(2)

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


def create_model():#학습 모델 만들기
    model = tf.keras.models.Sequential([
        #컨볼루션층
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(shape_1,shape_1,1),padding='same'),#3x3필터(w)- kernel을 32개 적용 후 relu 활성함수를 걸쳐 출력(relu : 0보다 큰값은 원값, 0보다 작으면 0)
        # Conv2D(필터의 개수, kernel 사이즈, strides=kernel의 이동량, activation=활성함수,input shape = 입력 모양,padding=이미지의 테두리에 padding을 두른다.) #패턴을 찾아낸다.(입력값ㄴ)
        tf.keras.layers.MaxPooling2D((2,2),padding='same'),#2x2에서 가장 큰값만 나온다(압축)
        tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D((2,2),padding='same'),
        tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        #Flatten층
        tf.keras.layers.Flatten(),
        #선형회귀층
        tf.keras.layers.Dense(128,activation='relu'),        
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    #모델 컴파일 
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #(optimizer='훈련과정(경사하강법 지정)', loss=손실함수,merix = 훈련을 모니터링하기 위해 사용)
    return model

model = create_model()#model 생성
model.summary() #모델 레이어층 확인


#저장 로드 위치
checkpoint_path = 'tensorflow/cp_face.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

#checkpoint callback 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)

#훈련하기
model.fit(train_imgs,y_train, epochs=50,callbacks=[cp_callback])#(트레이닝이미지, 레이블),validation_data=[test_imgs,y_test],callbacks=[cp_callback]

#정확도 테스트
# model.load_weights(checkpoint_path)
# test_loss, test_acc = model.evaluate(train_imgs,  y_train,verbose = 0)
#verbose	0 또는 1. 상세 모드. 0 = 자동, 1 = 진행 표시 줄
# test_loss, test_acc = model.evaluate(test_imgs,y_test,verbose = 2)
# print('\n테스트 정확도:', test_acc)



      
# predictions = model.predict(test_imgs)#이미지를 넣어 맞는지 확인

# print('test :',predictions[0].argmax())
# print('정답라벨:', y_test[3])
# cv2.imshow('ss',test_imgs[3])
# print(y_test)

# cv2.waitKey(0)
# print('\n테스트 정확도:', test_acc)







# 전체 모델을 HDF5 파일로 저장합니다
#model.save('my_model.h5')
# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
#new_model = keras.models.load_model('my_model.h5'


###가중치 저장 및 복원
# 가중치를 저장합니다
# model.save_weights('./checkpoints/my_checkpoint')

# # 가중치를 복원합니다
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')
