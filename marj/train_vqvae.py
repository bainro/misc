import os
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from vqvae import VQVAE
from torch import nn, optim
from scipy.ndimage import zoom
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Initialize model.
device = torch.device("cuda:0")
use_ema = True
model_args = {
    "in_channels": 1,
    "num_hiddens": 128, # number of feature maps / channels in some of the conv layers
    "num_downsampling_layers": 3, # conv layers (1st) with stride=2, so reducing the # of outputs
    "num_residual_layers": 3, # layers (2nd) with skip connections
    "num_residual_hiddens": 32, # number of feature maps / channels for the other conv layers
    "embedding_dim": 1,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)

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

# Initialize dataset.
transform = transforms.Compose([
   transforms.ToTensor(),
])

batch_size = 64
# Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation Learning".
beta = 0.25
lr = 3e-4

train_dataset = IXI_Dataset(f"./IXI_dataset_2d_hires/", transform=transform)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

mean = torch.zeros(1)
sq_mean = torch.zeros(1)

# Iterate over the dataset to compute mean and variance
num_samples = 0
for images in train_loader:
    # Update the mean (sum of values)
    mean += images.mean(dim=[0, 2, 3]).detach() * images.size(0)
    
    # Update the squared mean
    sq_mean += (images ** 2).mean(dim=[0, 2, 3]).detach() * images.size(0)
    
    num_samples += images.size(0)

# Final mean and squared mean
mean /= num_samples
sq_mean /= num_samples

# Compute variance (variance = E[X^2] - (E[X])^2)
variance = sq_mean - mean ** 2
train_data_variance = variance.to(device)

# Initialize optimizer.
train_params = [params for params in model.parameters()]
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

best_path = 'vqvae_morer_compressed.pth'
#'''
# Train model.
epochs = 30
eval_every = 100
best_train_loss = float("inf")
model.train()
for epoch in range(epochs):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = train_tensors.to(device)
        out = model(imgs)
        recon_error = criterion(out["x_recon"], imgs) / train_data_variance
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % eval_every) == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            total_train_loss /= n_train
            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss
                torch.save(model.state_dict(), best_path)

            print(f"total_train_loss: {total_train_loss}")
            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0
#'''

def save_img_tensors_as_grid(img_tensors, nrows, f):
    imgs = img_tensors.detach().cpu().numpy()
    imgs[imgs < -0.5] = -0.5
    imgs[imgs > 0.5] = 0.5
    imgs = 255 * (imgs + 0.5)
    (batch_size, img_size) = imgs.shape[:2]
    batch_size = imgs.shape[0]
    # assumes square images!
    img_size = imgs.shape[-1]
    ncols = batch_size // nrows
    if f == 'recon':
        img_arr = np.zeros((3, nrows * img_size, ncols * img_size))
    else:
        img_arr = np.zeros((1, nrows * img_size, ncols * img_size))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[:, row_start:row_end, col_start:col_end] = imgs[idx]
        if f == 'recon':
            # change from [ch,x,y] -> [x,y,ch]
            i = img_arr.transpose(1,2,0)
            Image.fromarray(i.astype(np.uint8), "RGB").save(f"{f}.jpg")
        else:
            Image.fromarray(img_arr[0].astype(np.uint8), "L").save(f"{f}.jpg")

model.load_state_dict(torch.load(best_path))    
# Generate and save reconstructions.
model.eval()

# @TODO test_loader with IndivsRobotic data
with torch.no_grad():
    for imgs in train_loader:
        break

    '''
    # for invidRobotics only
    imgs = nib.load("./1.nii.gz")
    imgs = imgs.get_fdata()
    imgs = imgs.transpose(2,0,1)
    imgs = torch.tensor(imgs).unsqueeze(0).float()

    # reshape T2 images to a constant, desired resolution
    # these are my target resolution values
    x_t = y_t = 256
    z_t = 128 # number of slices / channels
    # the real current resolution values
    z_r, x_r, y_r = imgs.shape[1:]
    x_m = x_t / x_r
    y_m = y_t / y_r
    z_m = z_t / z_r
    imgs = zoom(imgs, (1, z_m, x_m, y_m))
    imgs = torch.tensor(imgs.transpose(1,0,2,3)).float()

    imgs = imgs / imgs.max() # normalize between 0 & 1
    '''
    
    n_rows = 8
    save_img_tensors_as_grid(imgs, n_rows, "true")
    inference = model(imgs.to(device))["x_recon"]
    save_img_tensors_as_grid(inference, n_rows, "recon")
