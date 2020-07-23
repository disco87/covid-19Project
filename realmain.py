# -*- coding: utf-8 -*-
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import BooleanProperty, ListProperty, StringProperty, ObjectProperty
from kivy.core.window import Window
import func_module as fm
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
#리스트 생성 관련
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView

from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.utils import reify
from kivy.lang import Builder


Builder.load_string('''
# -*- coding: utf-8 -*-
#: import utils kivy
#: import os os
#: import Factory kivy.factory.Factory

<Bt@Button>
    font_name:'HANDotum'

<Manager>:
    screen_main: screen_main_id
    screen_one: screen_one_id    
    ScreenMain:
        id:screen_main_id
        name: 'ScreenMain'
    ScreenOne:
        id: screen_one_id
        name: 'Screen1'  
<ScreenMain>: 

    FloatLayout:  
        Bt:
            text: '노래듣기' 
            size_hint:0.2,0.1
            pos_hint:{'x':0.8,'y':0.4} 
            on_press: root.song_popup()        
        Bt:
            font_name:'HANDotum'
            text:'사진촬영'
            size_hint:0.2,0.1
            pos_hint:{'x':0.8,'y':0.5}
        Bt:
            text:'관리자 모드'            
            on_press: root.manager.current = 'Screen1'
            size_hint:0.2,0.1
            pos_hint:{'x':0.8,'y':0.6}
<SongPopup>:
    rv: rv
    id:pop
    size_hint:0.5,0.5
    auto_dismiss:False
    title:'MP3 Play List'
   
    FloatLayout:
        
        Bt:
            text:'X'            
            size_hint:10,10
            background_color:0,0,0,0
            pos_hint:{'x':-0.6,'y':-0.7}
            on_press:pop.dismiss()  
        Rv:
            pos_hint:{'x':0,'y':0.1}
            id:rv
            data:self.data
            viewclass: 'SelectableLabel'
            font_name:'HANDotum'
            SelectableRecycleBoxLayout:
                font_name:'HANDotum'
                default_size: None, dp(56)
                default_size_hint: 1, None
                size_hint_y: None
                height: self.minimum_height
                orientation: 'vertical'
                multiselect: False
                touch_multiselect: True
        ToggleButton:
            font_namm: 'HANDotum'
            text:'재생'
            
            group:'song'
            on_press:            
            size_hint:1, 0.1
            pos_hint:{'x':0,'y':0}
 
<SelectableLabel>:
    # Draw a background to indicate selection
    canvas.before:
        Color:
            rgba: (.0, .0, .2, .2) if self.selected else (0, 0, 0, 0)
        Rectangle:
            pos: self.pos
            size: self.size
    canvas.before:
        Color:
            rgba: (.0, .0, .2, .2) if self.selected else (0, 0, 0, 0)
        Rectangle:
            pos: self.pos
            size: self.size                
<ScreenOne>
    BoxLayout:
        orientation:'vertical'
    Button:
        text:'Main Page'
        on_press:root.manager.current='ScreenMain'        
''')



class ScreenMain(Screen,Image):#main page
    # pass
    def __init__(self, **kw):
        super(ScreenMain,self).__init__(**kw)
        self.fps = 24 #카메라 프레임 설정
        self.func = fm.Func_Class()
        Clock.schedule_interval(self.update,1.0/self.fps)              
       
    def update(self,dt):
        buf = self.func.live_show()        
        self.texture = buf
        

    def song_popup(self):#노래 리스트 팝업
        print (fm.Func_Class.file_list)
        song_pop = SongPopup()
        song_pop.open()    
        
class ScreenOne(Screen):#관리자 페이지
    pass

class SongPopup(Popup):
    rv = ObjectProperty()
    
    t_select = BooleanProperty(False)
    t_selectable = BooleanProperty(True)
    # def on_touch_down(self, touch):
    #     return super(SongPopup,self).on_touch_down(touch)
    pass
class Rv(RecycleView):
    def __init__(self, **kwargs):
        super(Rv, self).__init__(**kwargs)
        self.data = [{'text':str(i)} for i in fm.Func_Class.file_list]#data 를 만들 때 튜플 형식으로 만들어야 한다.(key:vlaue)

####################셀렉트 만들기########################
class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):# 셀렉트 리스트 화면 구성
    ''' Adds selection and focus behaviour to the view. '''


class SelectableLabel(RecycleDataViewBehavior, Label):#셀렉트 리스트가 동작 하는것을 감지 하는 클래스
    ''' Add selection support to the Label '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)
    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)
    
    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        ''' Respond to the selection of items in the view. '''        
        self.selected = is_selected
        if is_selected:
            print("selection changed to {0}".format(rv.data[index]))
        else:
            print("selection removed for {0}".format(rv.data[index]))
#########################################################

class Manager(ScreenManager):#스크린 전환을 위한 screen manager 
    screen_main_id = ObjectProperty()#각 스크린의 객체를 받기 위한 임시 오브젝트를 생성하여
    screen_one_id = ObjectProperty()#screenmain.kv에서 각 화면의 클래스를 할당해준다.


class Switch2App(App):
    def build(self):#kivy를 상속 받아 실행 했을 시 가장 처음에 실해후 실행 안함(init과 동일한 기능) - life cycle참조
        fm.Func_Class.cam_init()
        fm.Func_Class.song_init()
        return Manager()#manager 클래스를 실행하여 반환    

if __name__ == '__main__':
    Switch2App().run()
    