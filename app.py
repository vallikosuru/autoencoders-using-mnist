import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image

# Load decoder model
decoder = load_model("vae_decoder.h5")

# Function to generate a single image from z1 and z2
def generate_digit(z1, z2):
    z_sample = np.array([[z1, z2]])
    generated = decoder.predict(z_sample)
    image = generated[0].reshape(28, 28)
    return Image.fromarray((image * 255).astype(np.uint8))

# Function to create full latent space grid like Figure_1.png
def generate_latent_space_grid():
    n = 15  # Grid size
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # Grid range in latent space
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.tight_layout()

    # Save to temporary buffer
    plt.savefig("latent_grid.png")
    plt.close()

    return "latent_grid.png"

# Gradio interface
def generate(z1, z2):
    digit_image = generate_digit(z1, z2)
    latent_grid_path = generate_latent_space_grid()
    return digit_image, latent_grid_path

# Launch Gradio app
interface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Slider(-3, 3, step=0.1, label="Latent Dimension z1"),
        gr.Slider(-3, 3, step=0.1, label="Latent Dimension z2")
    ],
    outputs=[
        gr.Image(type="pil", label="Generated Digit"),
        gr.Image(type="filepath", label="Full Latent Space Grid")
    ],
    title="VAE Latent Space Digit Generator",
    description="Use the sliders to choose a 2D latent vector and generate a digit. Also view the full latent space grid!"
)

interface.launch()
