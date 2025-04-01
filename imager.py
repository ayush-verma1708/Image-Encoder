import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import os
from torchvision.utils import save_image  # PyTorch function to save image

# Placeholder for an image dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

# Example transformation for image pre-processing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Placeholder for a simple Encoder and Decoder (VAE-like architecture)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(128*32*32, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        z = self.fc2(x)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128*32*32)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 32, 32)
        x = torch.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))  # output image
        return x

# Define a simple model structure combining Encoder and Decoder
class MultiDecoderModel(nn.Module):
    def __init__(self):
        super(MultiDecoderModel, self).__init__()
        self.encoder = Encoder()
        self.decoder_lion = Decoder()  # Decoder for Lion
        self.decoder_penguin = Decoder()  # Decoder for Penguin

    def forward(self, x, style='lion'):
        z = self.encoder(x)
        if style == 'lion':
            return self.decoder_lion(z)
        elif style == 'penguin':
            return self.decoder_penguin(z)
        else:
            raise ValueError("Unknown style")

# Assuming image paths are already available
image_paths = ['1.jpg', '2.jpg']  # Modify as per your paths

# Create the dataset and data loader
dataset = ImageDataset(image_paths, transform)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model
model = MultiDecoderModel()

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in data_loader:
        batch = batch.cpu()  # Assuming using GPU
        optimizer.zero_grad()

        # Forward pass for lion style
        output_lion = model(batch, style='lion')
        loss_lion = loss_fn(output_lion, batch)

        # Forward pass for penguin style
        output_penguin = model(batch, style='penguin')
        loss_penguin = loss_fn(output_penguin, batch)

        # Combine losses (optional, could focus on one)
        total_loss = loss_lion + loss_penguin
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")

# Now, after training is done, let's generate and save some images
output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate images for a sample batch after training
with torch.no_grad():
    for batch in data_loader:  # Using the first batch for testing
        batch = batch.cpu()

        # Generate images for 'lion' and 'penguin' style
        output_lion = model(batch, style='lion')
        output_penguin = model(batch, style='penguin')

        # Clamp output to [0, 1] to avoid out-of-range values
        output_lion = torch.clamp(output_lion, 0, 1)
        output_penguin = torch.clamp(output_penguin, 0, 1)

        # Save the images
        save_image(output_lion, os.path.join(output_dir, "lion_output.png"))
        save_image(output_penguin, os.path.join(output_dir, "penguin_output.png"))

        break  # Exit after processing one batch

print("Images saved to 'generated_images' folder.")
