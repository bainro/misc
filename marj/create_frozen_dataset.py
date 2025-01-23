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
batch_size = 100 # needs to 100 or bigger!
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
    shuffle=True,
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
save_dir = f"./DogsCats100_npy"
os.makedirs(save_dir, exist_ok=True)
with torch.no_grad():
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
            f = os.path.join(save_dir, f"{i:03d}_{cat_or_dog}.npy")
            # The 2x28x28 output of the VQVAE's encoder!
            np.save(f, x[i])