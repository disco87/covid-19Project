import pygame
from pygame import mixer
import os


class FuncMp3():   
    def __init__(self):
        self.path_dir =  os.path.realpath('.') + '\mp3' #mp3 root 위치 변수
        FuncMp3.file_list = os.listdir(self.path_dir) #폴더내의 파일 list 생성
        # FuncMp3.file_list.sort() #리스트내 이름정렬
        mixer.init()#mixer init
        
    
    #노래 재생
    def song_play(self,path):   
        '''
        음악 재생
        path = 음악경로
        '''       
        mixer.music.load(self.path_dir + path)#파일 load
        mixer.music.play()#mp3 play
    def song_stop(self):
        mixer.music.stop()

if __name__ == '__main__':
    print('ahahahahahahhahahah')



