import gradio as gr
import numpy as np
import tensorflow as tf
import os
import pickle
from PIL import Image
from architecture.CVAE import CVAE

def load_from_file(fp='model_1.pickle'):
    with open('models/' + fp, 'rb') as f:
        return pickle.load(f)

def load_photos(img_dir='photos'):
    imgs = []
    for fp in os.listdir(img_dir):
        imgs.append(Image.open(os.path.join(img_dir, fp)))
    
    return (np.asarray(imgs) / 255.).astype('float32')

model = load_from_file()
train_images = load_photos()

def plot_latent_image(ai=0, bi=0, ci=0, di=0, ei=0, fi=0, gi=0, hi=0, ji=0):
    z = np.array([[ai, bi, ci, di, ei, fi, gi, hi, ji]])
    x_decoded = model.sample(z)
    img = tf.reshape(x_decoded[0], (256, 256, 3)).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))

def reset():
    return [0 for _ in range(9)]
    # return [gr.update(value=0) for _ in range(9)] # not working anymore ?!

with gr.Blocks() as demo:
    gr.Markdown('Sliders test')
    
    with gr.Row():
        with gr.Column(scale=1):
            p1 = gr.Slider(label='Parameter 1', minimum=-10, maximum=10, step=0.01, value=0)
            p2 = gr.Slider(label='Parameter 2', minimum=-10, maximum=10, step=0.01, value=0)
            p3 = gr.Slider(label='Parameter 3', minimum=-10, maximum=10, step=0.01, value=0)
            p4 = gr.Slider(label='Parameter 4', minimum=-10, maximum=10, step=0.01, value=0)
            p5 = gr.Slider(label='Parameter 5', minimum=-10, maximum=10, step=0.01, value=0)
            p6 = gr.Slider(label='Parameter 6', minimum=-10, maximum=10, step=0.01, value=0)
            p7 = gr.Slider(label='Parameter 7', minimum=-10, maximum=10, step=0.01, value=0)
            p8 = gr.Slider(label='Parameter 8', minimum=-10, maximum=10, step=0.01, value=0)
            p9 = gr.Slider(label='Parameter 9', minimum=-10, maximum=10, step=0.01, value=0)
            sliders = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
            
            btn_reset = gr.Button('Reset')
            btn_reset.click(fn=reset, outputs=sliders)
        
        with gr.Column(scale=2):
            out_img = gr.Image(label='Output', interactive=False)
            
            for slider in sliders:
                slider.change(fn=plot_latent_image, inputs=sliders, outputs=out_img)
            
            with gr.Row():
                btn_run = gr.Button('Run')
                btn_run.click(fn=plot_latent_image, inputs=sliders, outputs=out_img)

demo.launch()