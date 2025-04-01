import matplotlib.pyplot as plt
from PIL import Image

# Function to display an image
def display_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Generated Image: {image_path}")
    plt.axis('off')  # Turn off axes for a cleaner look
    plt.show()

# Display generated images
display_image("generated_images/lion_output.png")
display_image("generated_images/penguin_output.png")
