# Latent Skyline Explorer
![python version](https://img.shields.io/badge/python-v3.12.1-green?logo=python) ![build version](https://img.shields.io/badge/build-v0.1-blue)

Latent Skyline Explorer is a Gen-AI app with a graphical interface built using Kivy (or Gradio) that allows users to interact with a Convolutional Variational Autoencoder (Conv-VAE) model. Users can manipulate the latent space of the Conv-VAE to generate and visualize different images based on the model's training. The provided pre-trained model uses classic deconvolution, which produces a lot of visual artifacts. You can train a model using an upsampling & convolution approach in the decoder, which creates less artifacts but it is notoriously blurrier. A technique called PixelShuffle is supposed to be much better, but I have not managed to get it to work.

## Features

- **Latent Space Manipulation**: Use nine sliders to adjust the values of the latent vector and see the effect on the generated image in real-time.
- **Preset Loading**: Load preset latent vectors corresponding to specific training images by entering an index. (Kivy interface only)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Huraqan/Latent-Skyline-Explorer.git
    cd SkylineApp
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run inference on the trained model with the GUI application by either using the Gradio interface `python gui_gradio.py` or by using Kivy `python gui_kivy.py`. The latter has more complete functionality with preset loading.

To train a model on your own images, just replace the content of the `photos` folder with your own and go through the `VAE-Training.ipynb` notebook.

## Requirements

Python 3.12.1+, Kivy or Gradio, Numpy, TensorFlow, Tensorflow Probabilities, Pillow, Matplotlib

## License

This project is licensed under the MIT License. See the LICENSE file for details

## Acknowledgements

The Conv-VAE model is based on TensorFlow.

## Sources

- https://www.tensorflow.org/tutorials/generative/cvae?hl=en
- https://distill.pub/2016/deconv-checkerboard/