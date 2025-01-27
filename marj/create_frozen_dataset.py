import os
import torch
import numpy as np
from PIL import Image
from vqvae import VQVAE
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


torch.manual_seed(42)

# Initialize dataset.
batch_size = 100
train_dataset = datasets.ImageFolder('/home/rbain/Downloads/DogCats100', transform=transforms.Compose(
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
    shuffle=False,
    num_workers=8,
)

test_dataset = datasets.ImageFolder('/home/rbain/Downloads/DogCatsTest', transform=transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
    ]
))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 3,
    "num_residual_layers": 3,
    "num_residual_hiddens": 32,
    "embedding_dim": 2,
    "num_embeddings": 512,
    "use_ema": True,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)
model_path = 'vqvae_cvd.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    j = 0
    save_dir = f"./DogsCats100_npy"
    os.makedirs(save_dir, exist_ok=True)
    for (batch_idx, train_tensors) in enumerate(train_loader):
        imgs = train_tensors[0].to(device)
        labels = train_tensors[1].numpy()
        out = model(imgs)
        x = out["z_quantized"].detach().cpu().numpy()
        for i in range(x.shape[0]):
            if labels[i] == 0:
                cat_or_dog = "cat"
            else:
                cat_or_dog = "dog"
            # store label in filename
            f = os.path.join(save_dir, f"{j:03d}_{cat_or_dog}.npy")
            # The 2x28x28 output of the VQVAE's encoder!
            np.save(f, x[i])
            j += 1
    k = 0
    save_dir = f"./DogsCatsTest_npy"
    os.makedirs(save_dir, exist_ok=True)
    for (batch_idx, test_tensors) in enumerate(test_loader):
        imgs = test_tensors[0].to(device)
        labels = test_tensors[1].numpy()
        out = model(imgs)
        x = out["z_quantized"].detach().cpu().numpy()
        for i in range(x.shape[0]):
            if labels[i] == 0:
                cat_or_dog = "cat"
            else:
                cat_or_dog = "dog"
            # store label in filename
            f = os.path.join(save_dir, f"{k:03d}_{cat_or_dog}.npy")
            # The 2x28x28 output of the VQVAE's encoder!
            np.save(f, x[i])
            k += 1