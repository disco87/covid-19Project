from kivy.uix.image import Image
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.app import App

class MyButton(ToggleButtonBehavior, Image):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        # self.source = 'atlas://data/images/defaulttheme/checkbox_off'
    
    def on_state(self, widget, value):
        if value == 'down':
            print('aaa')
        else:
            print('bb')


class SampleApp(App):
    def build(self):
        return MyButton()


SampleApp().run()