
import tensorflow as tf
import pickle

@tf.keras.utils.register_keras_serializable()
class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder using classic deconvolution."""

    def __init__(self, latent_dim=9, **kwargs):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=16 * 16 * 256, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 256)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=2, padding='same',activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=3, kernel_size=3, strides=2, padding='same'),
            ]
        )
        
    def save_to_file(self, fp='model_1.pickle'):
        with open('models/' + fp, 'wb') as f:
            pickle.dump(self, f)
        print('\nFile saved as', fp, '\n')

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits






@tf.keras.utils.register_keras_serializable()
class CVAE2(tf.keras.Model):
    """Convolutional variational autoencoder using upsampling and convolution as decoding."""

    def __init__(self, latent_dim=9, **kwargs):
        super(CVAE2, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=16 * 16 * 256, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 256)),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
            ]
        )
    
    def save_to_file(self, fp='model_1.pickle'):
        with open('models/' + fp, 'wb') as f:
            pickle.dump(self, f)
        print('\nFile saved as', fp, '\n')

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits