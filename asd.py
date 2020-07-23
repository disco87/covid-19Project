import speech_recognition as sr
import pyaudio
import re
import pyglet
import threading
import os

# path = os.path.realpath('.') + '/gui'
# print('get',os.getcwd())
# print(path)
# name = input()


# with open(path + '/text/audio.txt', 'w') as ff:
#     print(name, 'dkjf',file = ff)
r = sr.Recognizer()
mic=sr.Microphone()


with mic as source:
    print('say1')
    audio = r.listen(source,phrase_time_limit=3)
print(r.recognize_google(audio,language='ko-KR'))

while 1:
    try:
        with mic as source:
            print('say~')                    
            # r.adjust_for_ambient_noise(source,duration=1)#노이즈 거르기
            audio=r.listen(source,phrase_time_limit=10)
        # text = open('C:/Users/w/Documents/python/gui/text/audio.txt','w')
        data=(r.recognize_google(audio,language='ko-KR'))
        print(data)
        # text.write('asdf')
        test = re.match('살려',data)
        print('test', test)
        aa = re.finditer('살려', data)
        print ('not:', not aa)
        print ('list',aa)
        print('data:',data)

        # print ('length : ', len(aa))
        if test != None:
            print ('구조요청중')
    except:
        print ('ng')
