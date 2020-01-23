import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, \
    Conv2DTranspose


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=32, name='encoder'):
        super().__init__(name=name)
        self.latent_dim = latent_dim

        self.conv1 = Conv2D(filters=32, kernel_size=3, strides=2,
                            activation='relu',
                            name='conv1')
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=2,
                            activation='relu', name='conv2')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(latent_dim * 2, name='dense')
        self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.flatten(inputs)
        inputs = self.dense(inputs)
        z_mean, z_log_var = tf.split(inputs, num_or_size_splits=2, axis=1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    def __init__(self, name='decoder'):
        super().__init__(name=name)
        self.dense = Dense(units=7 * 7 * 32, activation='relu', name='dense')
        self.reshape = Reshape(target_shape=(7, 7, 32), name='reshape')
        self.conv_trans_1 = Conv2DTranspose(filters=64,
                                            kernel_size=3,
                                            strides=(2, 2),
                                            padding='SAME',
                                            activation='relu',
                                            name='conv_trans_1')
        self.conv_trans_2 = Conv2DTranspose(filters=32,
                                            kernel_size=3,
                                            strides=(2, 2),
                                            padding='SAME',
                                            activation='relu',
                                            name='conv_trans_2')

        # sigmoid activation (get image with origianl size)
        self.conv_trans_3 = Conv2DTranspose(filters=1,
                                            kernel_size=3,
                                            strides=(1, 1),
                                            padding='SAME',
                                            activation='sigmoid',
                                            name='conv_trans_3')

    def call(self, inputs, **kwargs):
        inputs = self.dense(inputs)
        inputs = self.reshape(inputs)
        inputs = self.conv_trans_1(inputs)
        inputs = self.conv_trans_2(inputs)
        inputs = self.conv_trans_3(inputs)
        return inputs


class Sampling(tf.keras.layers.Layer):
    def __init__(self, name='sampling'):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, latent_dim, name='VAE'):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL divergence Loss
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
