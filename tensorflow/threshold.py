import cv2
import numpy as np
import os
import time
import dlib
import skimage
path = os.path.abspath('.') + '/tensorflow/trimg/1'
file_list = os.listdir(path)
# print (file_list)
cap = cv2.cv2.VideoCapture(0)#비디오 오픈
detector = dlib.get_frontal_face_detector()#dlib 얼굴 찾는 객체 생성
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#얼굴 이식 관련 data 파일 삽입


while 1:
    ret,frame = cap.read()
    img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    remap = []
    print (dets)
    for face in dets:
        cv2.rectangle(frame,(face.left(), face.top()), (face.right(), face.bottom()),(0,0,255),2)
        #얼굴 사각형
        crop_face = img_gray[face.top()-30:face.bottom() + 25, face.left()-30:face.right()+30]
        #얼굴 이미지 자르기
        shape = predictor(crop_face, face)#얼굴 landmark 찾기
        landmarks = np.array([[p.x,p.y]for p in shape.parts()])
        pts = np.array([[p[0],p[1]]for p in landmarks[0:27]])
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = crop_face[y:y+h,x:x+w].copy##########
        print(croped)
        print(rect)
        print(pts)
        # pts = pts - pts.min(axis = 0)
        # mask = np.zeros(croped.shape[:2], np.uint8)
        # cv2.drawContours(mask, [pts], -1, (255,255,255),-1,cv2.LINE_AA)
        # im = cv2.bitwise_and(croped, croped, mask=mask)
        # bg = np.ones_like(croped, np.uint8)*255
        # cv2.bitwise_not(bg,bg, mask=mask)
        # dst2 = bg+ im
        # cv2.imwrite("croped.png", croped)
        # cv2.imwrite("mask.png", mask)
        # cv2.imwrite("dst.png", im)
        # cv2.imwrite("dst2.png", dst2)

        # print(land)
        print(landmarks)
    
    cv2.imshow('test',frame)
    key = cv2.waitKey(0)
    if key == 27 :
        break
    if key == 32 :
        break


# import numpy as np
# import cv2

# img = cv2.imread("tensorflow/trImg/1/05-18-51-08.jpg")
# pts = np.array([[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]])
# print (pts)

# ## (1) Crop the bounding rect
# rect = cv2.boundingRect(pts)
# x,y,w,h = rect
# croped = img[y:y+h, x:x+w].copy()

# ## (2) make mask
# pts = pts - pts.min(axis=0)

# mask = np.zeros(croped.shape[:2], np.uint8)
# cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# ## (3) do bit-op
# dst = cv2.bitwise_and(croped, croped, mask=mask)

# ## (4) add the white background
# bg = np.ones_like(croped, np.uint8)*255
# cv2.bitwise_not(bg,bg, mask=mask)
# dst2 = bg+ dst


# cv2.imwrite("croped.png", croped)
# cv2.imwrite("mask.png", mask)
# cv2.imwrite("dst.png", dst)
# cv2.imwrite("dst2.png", dst2)


# def face_remap(shape):
#     remapped_image = cv2.convexHull(shape)
# return remapped_image





        # landmarks = np.array([[p.x,p.y]for p in shape.parts()])#landmark 포인트 리스트컴프레션
        # outline = landmarks[[*range(17), *range(26,16,-1)]]
        # Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
        # cropped_img = np.zeros(frame.shape, dtype=np.uint8)
        # cropped_img[Y, X] = frame[Y, X]
        # vertices = cv2.ConvexHull(landmarks).vertices
        # Y, X = skimage.draw.polygon(landmarks[vertices, 1], landmarks[vertices, 0])
        # cropped_img = np.zeros(frame.shape, dtype=np.uint8)
        # cropped_img[Y, X] = frame[Y, X]
        # cv2.imshow('test',cropped_img)
        # cv2.waitKey(0)