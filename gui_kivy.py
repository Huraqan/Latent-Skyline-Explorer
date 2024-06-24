import numpy as np
import tensorflow as tf
import os
import pickle
from PIL import Image as PILImage
from architecture.CVAE import CVAE

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.graphics.texture import Texture

if not __name__ == '__main__':
    exit()

def load_from_file(fp='model_1.pickle'):
    with open('models/' + fp, 'rb') as f:
        return pickle.load(f)

def load_photos(img_dir='photos'):
    imgs = []
    for fp in os.listdir(img_dir):
        imgs.append(PILImage.open(os.path.join(img_dir, fp)))
    
    return (np.asarray(imgs) / 255.).astype('float32') # Between 0.0 and 1.0

training_images = load_photos()
model = load_from_file()

Z_ORIGIN = np.array([[0 for _ in range(9)]])

def latent_image(z):
    x_decoded = model.sample(z)
    img = tf.reshape(x_decoded[0], (256, 256, 3)).numpy()
    return img

def preset_latent_vec(image):
    tensor = tf.data.Dataset.from_tensor_slices(image).batch(1).take(1).get_single_element()
    mean, logvar = model.encode(tensor)
    z = model.reparameterize(mean, logvar)
    return z

class SkylineApp(App):
    def build(self):
        # GLOBAL LAYOUT
        self.layout = BoxLayout(orientation='horizontal')
        
        ## LEFT SIDE LAYOUT
        self.menu_layout = BoxLayout(orientation='vertical', size_hint=(0.35, 1))
        
        ### SLIDERS LAYOUT
        self.slider_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.7))
        self.menu_layout.add_widget(self.slider_layout)
        self.sliders = []
        self.slider_texts = []
        
        for i in range(9):
            slider = Slider(min=-12, max=12, value=0, step=0.01, size_hint=(0.8, 0.1), height=2)
            slider.bind(value=self.update_image)
            self.sliders.append(slider)
            
            slider_text = TextInput(text='0', size_hint=(0.2, 0.1), input_filter='float')
            self.slider_texts.append(slider_text)
            
            slider_grid = GridLayout(cols=2)
            slider_grid.add_widget(slider)
            slider_grid.add_widget(slider_text)
            
            self.slider_layout.add_widget(slider_grid)
        
        # Reset sliders
        self.btn_reset = Button(text='Reset')
        self.btn_reset.bind(on_press=self.reset_sliders)
        self.slider_layout.add_widget(self.btn_reset)
        
        # Select preset
        self.preset_layout = GridLayout(cols=3)
        self.btn_decrease = Button(text='-', size_hint_x=None, width=40)
        self.btn_decrease.bind(on_press=self.decrease_preset)
        self.input_preset = TextInput(text='0', multiline=False, input_filter='int')
        self.btn_increase = Button(text='+', size_hint_x=None, width=40)
        self.btn_increase.bind(on_press=self.increase_preset)
        
        self.preset_layout.add_widget(self.btn_decrease)
        self.preset_layout.add_widget(self.input_preset)
        self.preset_layout.add_widget(self.btn_increase)
        
        self.btn_load = Button(text='Load preset', height=40)
        self.btn_load.bind(on_press=self.load_preset)
        
        self.slider_layout.add_widget(self.preset_layout)
        self.slider_layout.add_widget(self.btn_load)
        
        ### THUMBNAIL LAYOUT
        self.thumbnail = Image(size_hint=(1, 0.3), allow_stretch=True, keep_ratio=True)
        self.menu_layout.add_widget(self.thumbnail)
        
        ## LEFT SIDE FINISH
        self.layout.add_widget(self.menu_layout)
        
        ## RIGHT SIDE LAYOUT
        
        ### IMAGE LAYOUT
        self.image = Image(size_hint=(0.8, 1), allow_stretch=True, keep_ratio=True)
        
        self.layout.add_widget(self.image)
        
        self.update_image(None, None)
        self.update_thumbnail(None, None, preset = self.clip_index(self.input_preset.text))
        
        return self.layout
    
    def update_slider_text(self):
        for slider, text in zip(self.sliders, self.slider_texts):
            text.text = f'{slider.value:.3f}'
    
    def set_sliders(self, latent_values):
        for slider, value in zip(self.sliders, latent_values):
            slider.unbind(value=self.update_image)
            slider.value = value
            slider.bind(value=self.update_image)
    
    def reset_sliders(self, instance):
        self.set_sliders([0 for _ in range(9)])
        self.update_slider_text()
    
    def increase_preset(self, instance):
        current_value = int(self.input_preset.text)
        preset = self.clip_index(current_value + 1)
        self.input_preset.text = str(preset)
        self.update_thumbnail(None, None, preset)
    
    def decrease_preset(self, instance):
        current_value = int(self.input_preset.text)
        preset = self.clip_index(current_value - 1)
        self.input_preset.text = str(preset)
        self.update_thumbnail(None, None, preset)
    
    def clip_index(self, text):
        return np.clip(int(text), 0, training_images.shape[0] - 1)
    
    def load_preset(self, instance):
        preset = self.clip_index(self.input_preset.text)
        image = training_images[preset]
        self.display_image(image, self.thumbnail)
        
        z = preset_latent_vec([image])
        self.display_image(latent_image(z), self.image)
        
        latent_values = z.numpy().flatten().tolist()
        self.set_sliders(latent_values)
        self.update_slider_text()
    
    def update_image(self, instance, value):
        self.update_slider_text()
        
        latent_values = [slider.value for slider in self.sliders]
        z = np.array([latent_values])
        self.display_image(latent_image(z), self.image)
    
    def update_thumbnail(self, instance, value, preset):
        image = training_images[preset]
        self.display_image(image, self.thumbnail)
    
    def reset_thumbnail(self, instance, value):
        self.display_image(latent_image(Z_ORIGIN), self.thumbnail)

    def display_image(self, image_array, kivy_img):
        image_array = np.flip(image_array, axis=0)
        texture = Texture.create(size=(256, 256), colorfmt='rgb')
        texture.blit_buffer(image_array.tobytes(), colorfmt='rgb', bufferfmt='float')
        kivy_img.texture = texture


SkylineApp().run()