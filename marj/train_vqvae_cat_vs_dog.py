import torch
import numpy as np
from PIL import Image
from vqvae import VQVAE
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42)

def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    print(f"img_tensors.shape: {img_tensors.shape}")
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    print(f"batch_size: {batch_size}")
    print(f"img_size: {img_size}")
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * img_size, ncols * img_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")


# Initialize model.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 3,
    "num_residual_layers": 3,
    "num_residual_hiddens": 32,
    "embedding_dim": 2,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)

# Initialize dataset.
batch_size = 64
train_dataset = datasets.ImageFolder('/home/rbain/Downloads/DogCats', transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
    ]
))

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
for images, _labels in train_loader:
    # Update the mean (sum of values)
    mean += images.mean().detach() * images.size(0)
    
    # Update the squared mean
    sq_mean += (images ** 2).mean(dim=[]).detach() * images.size(0)
    
    num_samples += images.size(0)

# Final mean and squared mean
mean /= num_samples
sq_mean /= num_samples

# Compute variance (variance = E[X^2] - (E[X])^2)
variance = sq_mean - mean ** 2
train_data_variance = variance.to(device)

# Multiplier for commitment loss.
beta = 0.25

# Initialize optimizer.
train_params = [params for params in model.parameters()]
lr = 3e-4
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

best_path = 'vqvae_cvd.pth'
#'''
# Train model.
epochs = 250
batch_per_epoch = len(train_dataset) // batch_size
eval_every = min(100, batch_per_epoch)
best_train_loss = float("inf")
model.train()
for epoch in range(epochs):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = train_tensors[0].to(device)
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
            
model.load_state_dict(torch.load(best_path))    
# Generate and save reconstructions.
model.eval()

batch_size = 64
train_dataset = datasets.ImageFolder('/home/rbain/Downloads/DogCats100', transform=transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
    ]
))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

with torch.no_grad():
    for imgs, labels in train_loader:
        break
    
    n_rows = 8
    save_img_tensors_as_grid(imgs, n_rows, "true")
    inference = model(imgs.to(device))["x_recon"]
    save_img_tensors_as_grid(inference, n_rows, "recon")