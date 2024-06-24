# SkylineApp
![python version](https://img.shields.io/badge/python-v3.12.1-green?logo=python) ![build version](https://img.shields.io/badge/build-v0.1-blue)

SkylineApp is a graphical interface built using Kivy that allows users to interact with a Convolutional Variational Autoencoder (Conv-VAE) model. Users can manipulate the latent space of the CVAE to generate and visualize different images based on the model's training. The provided pre-trained model uses classic deconvolution, which has a lot of visual artifacts. One can train a model using an upsampling & convolution approach in the decoder, although it creates less artifacts it is notoriously blurrier.

## Features

- **Load Pre-trained Model**: Load a pre-trained CVAE model.
- **Latent Space Manipulation**: Use nine sliders to adjust the values of the latent vector and see the effect on the generated image in real-time.
- **Preset Loading**: Load preset latent vectors corresponding to specific training images by entering an index.

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

Run the application either using Gradio interface: `python gui_gradio.py`
Or using Kivy: `python gui_kivy.py`

## Requirements

Python 3.12.1+
Kivy or Gradio
Numpy
TensorFlow & Tensorflow Probabilities
Pillow
Matplotlib

## License
This project is licensed under the MIT License. See the LICENSE file for details

## Sources
https://www.tensorflow.org/tutorials/generative/cvae?hl=en
https://distill.pub/2016/deconv-checkerboard/