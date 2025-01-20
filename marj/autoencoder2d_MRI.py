import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchio as tio
import nibabel as nib
from torchvision import io
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (B, 3, H, W) -> (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Downsampling by 2 (B, 32, H/2, W/2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # Downsampling by 2 (B, 64, H/4, W/4)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Downsampling by 2 (B, 128, H/8, W/8)
            #nn.Linear(128*16*16, 100)
        )
        
        # Decoder: Transposed Convolutional layers (deconvolution)
        self.decoder = nn.Sequential(
            #nn.Linear(100, 128*16*16)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 128, H/8, W/8) -> (B, 64, H/4, W/4)
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 64, H/4, W/4) -> (B, 32, H/2, W/2)
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, H/2, W/2) -> (B, 3, H, W)
            nn.Sigmoid()  # Sigmoid activation to get output in [0, 1] range for image pixel values
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

transform = transforms.Compose([
   transforms.ToTensor(),
])

class IXI_Dataset(Dataset):
    def __init__(self, nifti_dir, transform=None):
        self.nifti_dir = nifti_dir
        self.transform = transform
        
        # Get all .png file paths in the directory
        _ = os.listdir(nifti_dir)
        self.file_paths = [os.path.join(nifti_dir, f) for f in _ if f.endswith('.png')]
    
    def __len__(self):
        # Return the number of samples (NIfTI files) in the dataset
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # grayscale (i.e. 1 channel)
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        
        return img

ixi_dataset = IXI_Dataset(f"./IXI_dataset_2d/", transform=transform)

train_loader = DataLoader(ixi_dataset, batch_size=256, shuffle=True, num_workers=8)

# Initialize model, loss function, and optimizer
model = Autoencoder()
device = 'cuda:0'
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train the model
best_loss = 1e8
best_path = 'autoencoder.pth'
num_epochs = 100
for epoch in range(num_epochs):
    for i, images in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # save the best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), best_path)
             
model.load_state_dict(torch.load(best_path))
model.eval()
with torch.no_grad():
    for images in train_loader:
        outputs = model(images.to(device))
        break

def plot_images(inputs, outputs):
    batch_size, _ch, _x, _y = outputs.shape

    if batch_size > 12:
        batch_size = 12 # more gets too crowded

    fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))
    if batch_size == 1:
        axes = [axes]
    for i in range(batch_size):
        # Show original images (first row)
        axes[0, i].imshow(inputs[i,0,:,:].cpu().detach().numpy(), cmap='gray')
        axes[0, i].axis('off')  # Hide axes for cleaner visualization
        axes[0, i].set_title(f"Original {i+1}")
        # Show reconstructed images (second row)
        o = outputs[i,0,:,:].cpu().detach().numpy()
        axes[1, i].imshow(o, cmap='gray')
        axes[1, i].axis('off')  # Hide axes for cleaner visualization
        axes[1, i].set_title(f"Reconstructed {i+1}")

    plt.tight_layout()
    plt.show()

plot_images(images, outputs)
