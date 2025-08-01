import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))

latent_inputs = keras.Input(shape=(2,))

x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

decoder.load_weights("vae_decoder.h5")

def decode_latent_vector(decoder, z):
    """
    Decodes a single latent vector z (shape: (latent_dim,))
    and returns the decoded image (shape: (28, 28)).
    """
    z = np.array(z).reshape(1, -1)
    x_decoded = decoder.predict(z, verbose=0)
    image = x_decoded[0].reshape(28, 28)
    return image
    
    # Example latent vector
z = [0.6, 0.9]
decoded_img = decode_latent_vector(decoder, z)

plt.imshow(decoded_img, cmap="gray")
plt.axis("off")
plt.show()


def plot_latent_space(decoder, n=15, figsize=15):
    scale = 2.0
    figure = np.zeros((28 * n, 28 * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(28, 28)
            figure[i * 28: (i + 1) * 28,
                   j * 28: (j + 1) * 28] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap="Greys_r")
    plt.axis("Off")
    plt.show()
plot_latent_space (decoder)