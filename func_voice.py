import os
import speech_recognition as sr
import pyaudio
import re
import pyglet
import threading
import time
import pygame
from pygame import mixer
from multiprocessing import Process



class FuncVoice:
    def __init__(self):
        print('init')
        self.chk = 0
        # self.t1 = threading.Thread(target=self.voice_live)
        # self.t1.daemon = True
        # self.t1.start()
        
        

    def voice_live(self):
    # a= 1

        r = sr.Recognizer()
        mic=sr.Microphone()
        while 1:
            try:
                with mic as source:
                    print('say~')                    
                    # r.adjust_for_ambient_noise(source,duration=1)#노이즈 거르기
                    audio=r.listen(source,phrase_time_limit=5)
                # text = open(path + '/gui/text/audio.txt','w')
                data=(r.recognize_google(audio,language='ko-KR'))
                # text.write(data)
                print (data)
                
                aa = re.finditer('살려', data)
                f = re.match('야호',data)
                # if f == None:
                print (f)
                print(data)
                if f != None:
                    print ('구조요청중')
                    self.chk = 1 
                else:
                    pass
                if self.chk == 1:    
                    print('1111111111')
                    # pygame.mixer.music.play()
                    # clock = pygame.time.Clock()
                    # while pygame.mixer.music.get_busy():
                    #     clock.tick(5)
                    # pygame.mixer.quit()
                elif self.chk == 2:
                    mixer.music.stop()
                    self.chk = 0
                else:
                    pass
            except:  
                print('예외발생')

            

# if __name__ == '__main__':
#     print ('이것도 되나요??안되나요???')
#     proc = Process(voice_live)