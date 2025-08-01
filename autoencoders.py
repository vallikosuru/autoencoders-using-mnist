from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize data to [0, 1]
x_train_norm = x_train.astype("float32") / 255.0
x_test_norm = x_test.astype("float32") / 255.0

# Define Autoencoder architecture
input_img = Input(shape=(28, 28))
x = Flatten()(input_img)
encoded = Dense(128, activation='relu')(x)
latent = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(latent)
decoded = Dense(784, activation='sigmoid')(decoded)
output_img = Reshape((28, 28))(decoded)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train autoencoder
autoencoder.fit(
    x_train_norm, x_train_norm,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test_norm, x_test_norm)
)

# ✅ Save the model to 'model/autoencoder.h5'
os.makedirs("model", exist_ok=True)
autoencoder.save("model/autoencoder.h5")

# Encode and decode some digits
decoded_imgs = autoencoder.predict(x_test_norm)

# Number of digits to display
n = 10

plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_norm[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
