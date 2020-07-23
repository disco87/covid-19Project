 #-*- coding: utf-8 -*- 
import PIL
from PIL import Image,ImageTk,ImageDraw,ImageFont
import cv2
import os
import pygame
from pygame import mixer
import numpy as np
import dlib
from kivy.graphics.texture import Texture
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import speech_recognition as sr
import pyaudio
import re
import pyglet

class Func_Class:  
    def __init__(self):
        Func_Class.chk = 0
        self.m = self.create_model()       
        checkpoint_path = 'C:/aw/python/pro03/tensorflow/cp_face.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path) 
        self.m.load_weights(checkpoint_path)
        self.name_class = ['김영수','남상민','정효균']



    
#https://github.com/opencv/opencv/tree/master/data/haarcascades xml페이지 - 라이브러리  
###카메라 관련 처리######################################    
    def cam_init(cls):#카메라 관련 초기화
        Func_Class.cap = cv2.VideoCapture(0)
        Func_Class.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 450)
        Func_Class.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        Func_Class.detector = dlib.get_frontal_face_detector()
        # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
        Func_Class.predictor = dlib.shape_predictor('C:/aw/python/pro03/shape_predictor_68_face_landmarks.dat')
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
        # 
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
            n_predict = 0
            for face in dets:                
                shape = Func_Class.predictor(frame, face) #얼굴에서 68개 점 찾기
                #############################################################
                ####################이미지 전처리 & 얼굴 찾기##################                
                try: 
                    chk_img = img_gray[face.top()-10:face.bottom() + 15, face.left()-20:face.right()+10]
                    im = cv2.resize(chk_img,dsize=(28,28),interpolation=cv2.INTER_LINEAR)
                    im = cv2.GaussianBlur(im,(5,5),0)#노이즈 제거를 위해 가오시안 블러처리 
                    x_train = np.array(im)
                    x_train = x_train/255
                    train_imgs = x_train.reshape(1,28,28,1) 
                    predictions = self.m.predict(train_imgs)
                    n_predict = self.m.predict(train_imgs)
                    print (np.argmax(predictions[0]))#찾은 라벨의 맥스값을 출력
                    print(predictions[0])

                    # print('check',tf.argmax(predictions,1))
                except Exception as e:
                    print(e)
                ##################################################
                list_points = []
                for p in shape.parts():
                    list_points.append([p.x, p.y])
                list_points = np.array(list_points)#배열형태로 바꾸어준다.
                for i,pt in enumerate(list_points[index]):#각 지정한 포인트에 맞게 랜드마크 점을 찾는다(밑에서 찾을 것을 지정해줌)
                    pt_pos = (pt[0], pt[1])                    
                    cv2.circle(frame, pt_pos, 2, (0, 150, 0), -1)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                    (0, 0, 255), 3)
                try:
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(((face.left() + face.right())/2 - 10, face.top() - 25), self.name_class[np.argmax(n_predict[0])-1], font = ImageFont.truetype("C:/aw/python/pro03/gui/HiMelody-Regular.ttf", 25), fill = (0, 255, 0, 0))
                    frame = np.array(img_pil)                        
                except Exception as e:
                    pass
                

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)#color 배열 변경
            cv2image = cv2.flip(cv2image,0)
            # cv2image = cv2.flip(cv2image,1)
            buf = cv2image.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgba')
            image_texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')            
            return image_texture

    def create_model(self):#학습 모델 만들기
        model = tf.keras.models.Sequential([
            #컨볼루션층
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1),padding='same'),#3x3필터(w)- kernel을 32개 적용 후 relu 활성함수를 걸쳐 출력(relu : 0보다 큰값은 원값, 0보다 작으면 0)
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
        # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        #(optimizer='훈련과정(경사하강법 지정)', loss=손실함수,merix = 훈련을 모니터링하기 위해 사용)
        return model

    # 카메라 이진화 변경
    def cam_pic(self):
        ret, frame = Func_Class.cap.read()
        frame = cv2.flip(frame,0)
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
###노래 관련############################################
    #노래 관련 초기화 
    def song_init(cls): 
        Func_Class.path_dir ='C:/aw/python/pro03/mp3/' #mp3 root 위치 변수
        Func_Class.file_list = os.listdir(Func_Class.path_dir) #폴더내의 파일 list 생성
        Func_Class.file_list.sort() #리스트내 이름정렬
        mixer.init()#mixer init
        
    song_init = classmethod(song_init)
    #노래 재생
    def song_play(self,path):          
        mixer.music.load(Func_Class.path_dir + path)#파일 load
        mixer.music.play()#mp3 play
    def song_stop(self):
        mixer.music.stop()
    def voice(self):
        
        r = sr.Recognizer()#구글 접속(음성 분석)
        mic=sr.Microphone()#마이크 조인


        

        try:
            with mic as source:
                print('say~')
                audio=r.listen(source,phrase_time_limit=3)
            text = open('C:/aw/python/pro03/audio/audio.txt','w')
            data=(r.recognize_google(audio,language='ko-KR'))
            print(data)
            text.write(data)
            if not re.finditer('살려', data):
                pass
                # Func_Class.chk = 0
            else:
                print ('구조요청중')
                Func_Class.chk = 1                
                
            if Func_Class.chk == 1:    
                pygame.mixer.music.load("C:/aw/python/pro03/audio/m2.mp3")
                pygame.mixer.music.play()
                clock = pygame.time.Clock()
                while pygame.mixer.music.get_busy():
                    clock.tick(5)
                pygame.mixer.quit()
            elif Func_Class.chk == 2:
                mixer.music.stop()
                Func_Class.chk = 0
            else:
                pass
            text.close()
        except:  
            print('예외발생')
        return Func_Class.chk