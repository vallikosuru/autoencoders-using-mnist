# 🧠 MNIST-VAE-Deep-Learning

This is a Deep Learning project that demonstrates how to build a **Variational Autoencoder (VAE)** using **TensorFlow and Keras**. The model is trained on the **MNIST dataset**, which contains images of handwritten digits (0 to 9). After training, the VAE can learn to compress images into a smaller representation (called latent space) and then generate new, realistic-looking digit images.

---

## 📌 What is This Project?

This project builds and trains a **Variational Autoencoder** on the MNIST dataset to:
- Learn compressed features of digits (in a 2D latent space)
- Reconstruct the original digit images
- Generate entirely new digit images from random latent vectors

This helps you understand how **deep generative models** work and how VAEs can generate new data.

---

## 🛠️ Technologies Used

| Tool / Library    | Purpose                             |
|------------------|-------------------------------------|
| Python           | Programming language                |
| TensorFlow / Keras | Deep learning model implementation |
| NumPy            | Mathematical operations             |
| Matplotlib       | Plotting and visualization          |
| MNIST Dataset    | Handwritten digits for training     |

---

## 🧠 What is a Variational Autoencoder (VAE)?

A **Variational Autoencoder** is a type of neural network used to generate new data points. It consists of:
- **Encoder**: Compresses input data into a smaller **latent representation** (e.g., 2D vector)
- **Decoder**: Takes the latent vector and reconstructs the original image
- **Latent Space**: A space where similar digits are placed closer together

Instead of learning a fixed encoding, VAE learns a **distribution (mean and variance)** from which we can sample new latent vectors to generate new images.

---

## 📊 Example Results

### 🔹 1. Original vs Reconstructed Digits
The VAE learns to reconstruct digits. After training:

| Original Image | Reconstructed Image |
|----------------|---------------------|
| ![](images/original.png) | ![](images/reconstructed.png) |

### 🔹 2. Latent Space Visualization
By mapping latent values across a 2D space, we can visualize how the model generates different digits.

| Latent Space Grid |
|-------------------|
| ![](images/latent_space_grid.png) |

---

