 #-*- coding: utf-8 -*- 
import PIL
from PIL import Image,ImageTk
import cv2
import os
from pygame import mixer
import numpy as np
import dlib
from kivy.graphics.texture import Texture
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from PIL import Image,ImageTk,ImageDraw,ImageFont


class Func_Class:
#https://github.com/opencv/opencv/tree/master/data/haarcascades xml페이지 - 라이브러리  
###카메라 관련 처리######################################    
    def cam_init(cls):#카메라 관련 초기화
        Func_Class.cap = cv2.VideoCapture(0)
        Func_Class.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
        Func_Class.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        Func_Class.detector = dlib.get_frontal_face_detector()
        # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
        Func_Class.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성 
    cam_init = classmethod(cam_init)     
    def off_show(cls):
        cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Func_Class.cap.release()
        cv2.destroyAllWindows()
    off_show = classmethod(off_show)
    #카메라 컬러 출력 
    def live_show(self):              
        ret, frame = Func_Class.cap.read()
        checkpoint_path = 'tensorflow/cp_face.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if ret: 
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = Func_Class.detector(img_gray, 1)
            ALL = list(range(0, 68)) 
            RIGHT_EYEBROW = list(range(17, 22)) # 눈썹 17 - 21
            LEFT_EYEBROW = list(range(22, 27)) #눈썹 22 - 26
            RIGHT_EYE = list(range(36, 42))  
            LEFT_EYE = list(range(42, 48))  
            NOSE = list(range(27, 36))  
            MOUTH_OUTLINE = list(range(48, 61))  
            MOUTH_INNER = list(range(61, 68)) 
            JAWLINE = list(range(0, 17)) #턱 0 - 16
            index = ALL
            for face in dets:
                shape = Func_Class.predictor(frame, face) #얼굴에서 68개 점 찾기
                ####이미지 전처리###########
                chk_img = img_gray[face.top()-10:face.bottom() + 15, face.left()-20:face.right()+10]
                blur = cv2.GaussianBlur(chk_img,(5,5),0)#노이즈 제거를 위해 가오시안 블러처리 
                blur = cv2.resize(blur,dsize=(28,28),interpolation=cv2.INTER_LINEAR)  
                x_train = np.array(blur)
                train_imgs = x_train.reshape(1,28,28,1)
                m = self.create_model()
                m.load_weights(checkpoint_path)
                predictions = m.predict(train_imgs)
                print(tf.argmax(predictions,1).eval())
                ####################
                list_points = []
                for p in shape.parts():
                    list_points.append([p.x, p.y])
                list_points = np.array(list_points)#배열형태로 바꾸어준다.
                for i,pt in enumerate(list_points[index]):#각 지정한 포인트에 맞게 랜드마크 점을 찾는다(밑에서 찾을 것을 지정해줌)
                    pt_pos = (pt[0], pt[1])                    
                    cv2.circle(frame, pt_pos, 2, (0, 150, 0), -1)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                    (0, 0, 255), 3)
                
    
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)#color 배열 변경z
            buf = cv2image.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgba')
            image_texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte') 
            return image_texture
    # 카메라 이진화 변경
    def cam_pic(self):
        ret, frame = Func_Class.cap.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
        buf_pic = img.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgba')
        image_texture.blit_buffer(buf_pic, colorfmt='rgba', bufferfmt='ubyte') 
        return image_texture


    def live_black(self): 
        _, frame = Func_Class.cap.read()            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#gray test
        ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)#gray test
        img = PIL.Image.fromarray(dst)#gray test
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk


    def create_model(self):#학습 모델 만들기
        model = tf.keras.models.Sequential([
            #컨볼루션층
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(shape_1,shape_1,1),padding='same'),#3x3필터(w)- kernel을 32개 적용 후 relu 활성함수를 걸쳐 출력(relu : 0보다 큰값은 원값, 0보다 작으면 0)
            # Conv2D(필터의 개수, kernel 사이즈, strides=kernel의 이동량, activation=활성함수,input shape = 입력 모양,padding=이미지의 테두리에 padding을 두른다.) #패턴을 찾아낸다.(입력값ㄴ)
            tf.keras.layers.MaxPooling2D((2,2),padding='same'),#2x2에서 가장 큰값만 나온다(압축)
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D((2,2),padding='same'),
            tf.keras.layers.Conv2D(128,(3,3), activation='relu'),
            #Flatten층
            tf.keras.layers.Flatten(),
            #선형회귀층
            tf.keras.layers.Dense(128,activation='relu'),        
            tf.keras.layers.Dense(10,activation='softmax')
        ])
        #모델 컴파일 
        #model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        #(optimizer='훈련과정(경사하강법 지정)', loss=손실함수,merix = 훈련을 모니터링하기 위해 사용)
        return model
    def picture(self):
        _, frame = Func_Class.cap.read()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = time.strftime('%m-%d-%M-%S', time.localtime(time.time()))
        imsi = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(imsi, cv2.COLOR_BGR2GRAY)
        dets = Func_Class.detector(img_gray, 1) # 얼굴 찾기

        # index = list(range(0,27))#턱선 포인트와 양쪽 눈썹 포인트 위치
        # point_arr = [] #여러명의 얼굴 이미지의 좌표를 저장하기위함.
        img_arr = [] #여러명의 이미지를 넣기 위함

        
        for k, face in enumerate(dets): #k : 검출 개수 , 검출 좌표 - 얼굴을 찾은 만큼 반복
            shape = Func_Class.predictor(img_gray, face) #얼굴에서 좌표 찾기(눈썹, 턱, 눈 등)
            # cv2.rectangle(img_gray,(0,50),(face.right(),face.bottom()),(0,0,255),2)
            img_arr.append(img_gray[face.top()-10:face.bottom() + 15, face.left()-20:face.right()+10])#검출한 이미지 리스트에 넣기
          
            # imsi_arr = []
            # for k,p in enumerate(shape.parts()):
            #     if k > len(index):
            #         break
            #     imsi_arr.append([p.x,p.y])#shape에서 찾은 좌표를 imsi_arr 하나씩 추가
            #     hull = cv2.convexHull([p.x,p.y], clockwise=False) 
            #     cv2.drawContours(img_gray, hull,0, (0, 0, 255), 2)                
            #     print(k)  
            # imsi_arr = np.array(imsi_arr)#np type의 배열 형대토 바꾸어 준다.
            # point_arr.append(imsi_arr[0,0]) #첫번째 인덱스는 각 사람[찾은위치,좌표데이터][위치,자표]                     
            #print (point_arr[0][1]) #첫번째 인덱스는 찾은 위치 두번째 인덱스튼 0:x좌표 1은 y좌표
        cv2.imshow('live',frame)
        if not img_arr:
            pass 
            # cv2.imshow(t,img_gray)
            # print(img_gray.shape)
        else:
            for face in img_arr:
                blur = cv2.GaussianBlur(face,(5,5),0)#노이즈 제거를 위해 가오시안 블러처리 
                try:
                    re_img = cv2.resize(blur,dsize=(28,28),interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    pass
                                
                # th = cv2.adaptiveThreshold(re_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#인계점 찾기                
                # cv2.imshow(t,re_img) 
                # key = cv2.waitKey(0)
                # cv2.imwrite('tensorflow/3/'+t+'.jpg', re_img)#저장
                # return blur
                # print ('faces')
                # return re_img
        
        # b,g,r,a = 0,255,0,0
        # fontpath = "C:/Users/w/Documents/python/gui/HiMelody-Regular.ttf" # <== 这里是宋体路径 
        # font = ImageFont.truetype("C:/Users/w/Documents/python/gui/HiMelody-Regular.ttf", 32)
        # img_pil = Image.fromarray(frame)
        # draw = ImageDraw.Draw(img_pil)
        # draw.text((50, 80),  "김영수김영수김영수", font = ImageFont.truetype("C:/Users/w/Documents/python/gui/HiMelody-Regular.ttf", 32), fill = (b, g, r, a))
        # frame = np.array(img_pil)

        cv2.imshow('test',frame)
   
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()            
            # return
        else :
            cv2.imwrite('tensorflow/trimg/2/' + t +'.jpg',re_img)
            # cv2.destroyAllWindows()
            self.picture()
            



if __name__ == '__main__':
    f = Func_Class()
    f.cam_init()
    f.picture()

#https://stackoverrun.com/ko/q/12806463
#    def face_remap(shape):
#           remapped_image = cv2.convexHull(shape)
#       return remapped_image
#https://stackoverrun.com/ko/q/11019283
#좌표 https://blog.naver.com/PostView.nhn?blogId=chandong83&logNo=221506107511&categoryNo=29&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView


