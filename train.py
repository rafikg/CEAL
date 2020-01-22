import tensorflow as tf
from . import VAE

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images, 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images, 28, 28, 1).astype('float32')

# Normalizing the images
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

# tf.data to create batches and shuffle dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
    TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(
    TEST_BUF).batch(BATCH_SIZE)

latent_dim = 64

vae_model = VAE(latent_dim=64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

his = vae_model.fit(X=train_dataset, validation_data=)

