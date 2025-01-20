import os
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import DataLoader
from torchvision.utils import save_image


transforms = [
    tio.ToCanonical(), # to RAS
]

# 578 T2-weighted MRI images :)
ixi_dataset = tio.datasets.IXI(
    'path/to/ixi_root/',
    modalities=['T2'],
    transform=tio.Compose(transforms),
    download=True,
)

train_loader = DataLoader(ixi_dataset, batch_size=1, shuffle=False, num_workers=8)

save_dir = f"./IXI_dataset_2d_hires/"
os.makedirs(save_dir, exist_ok=True)
for i, data in enumerate(train_loader):
    # before squeeze shape: [1, 1, 128, 256, 256]
    T2 = data['T2']['data'].squeeze(0)
    # each subject has 128 slices in their T2
    for j in range(128): # from customized tio's image.py
        T2_slice = T2[:,j,:,:]
        path = os.path.join(save_dir, f"{i:03d}_s{j:02d}.png")
        # T2 max is the maximum voxel intensity for each subject
        save_image(T2_slice / T2.max(), path)

