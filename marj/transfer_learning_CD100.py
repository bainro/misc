import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)
        return torch.relu(h)

class EncoderTail(nn.Module):
    def __init__(self):
        super(EncoderTail, self).__init__()

        self.residual_stack = ResidualStack(
            num_hiddens=2, num_residual_layers=3, num_residual_hiddens=32
        )

        self.tail = nn.Sequential(
            # 2,28,28 -> 4,14,14
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 4,14,14 -> 8,7,7
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 8,7,7 -> 12,4,4
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 12,4,4 -> 16,2,2
            nn.Conv2d(12, 16, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),

            # 16,2,2 -> 1,1,1
            nn.Conv2d(16, 1, kernel_size=2, stride=2, padding=0),
        )

    '''
        # trained on big test, eval'd on 100 (so reversed)
            ### Attempt #1: Fully Convolutional Model (68%)
            # 2,28,28 -> 4,14,14
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 4,14,14 -> 8,7,7
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 8,7,7 -> 12,4,4
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 12,4,4 -> 16,2,2
            nn.Conv2d(12, 16, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),

            # 16,2,2 -> 1,1,1
            nn.Conv2d(16, 1, kernel_size=2, stride=2, padding=0),

            ### Attempt #2: MLP (55%)
            nn.Linear(2*28*28, 50),
            nn.Linear(50, 1),

            ### Attempt #3: Attempt #1 with an initial 3 residual blocks (76%)
            ### Attempt #4: Attempt #3 with 6 residual blocks (84/82/81%) (2,6,32)
    '''

    def forward(self, x):
        x = self.residual_stack(x)
        return self.tail(x)

class CD100_NPY_Dataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.npy_dir = npy_dir
        self.transform = transform
        
        # Get all .png file paths in the directory
        _ = os.listdir(npy_dir)
        self.file_paths = [os.path.join(npy_dir, f) for f in _ if f.endswith('.npy')]
    
    def __len__(self):
        # Return the number of samples (npy files) in the dataset
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        x_path = self.file_paths[idx]    
        x = np.load(x_path)
        y = 1 if x_path[-7:-4] == "cat" else 0
        return x, y # i.e. data, label

train_dataset = CD100_NPY_Dataset(f"./DogsCats100_npy/")
test_dataset = CD100_NPY_Dataset(f"./DogsCatsTest_npy/")

#'''
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
train_loader_eval = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8)
'''
train_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=8)
test_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
'''

# Initialize model, loss function, and optimizer
model = EncoderTail().to(device)

train_params = [params for params in model.parameters()]
optimizer = optim.Adam(train_params, lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

best_path = 'model.pth'
# Train model.
epochs = 450
best_loss = float("inf")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        x = train_tensors[0].to(device)
        labels = train_tensors[1].to(device)
        out = model(x).squeeze(-1).squeeze(-1)
        #print(f"guess: {out.detach().cpu()} label: {labels.detach().cpu()}")
        loss = criterion(out, labels.view(-1, 1).float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

    print(f"epoch: {epoch}", flush=True)
    print(f"total_train_loss: {total_loss}")
    total_loss /= n_train
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), best_path)

    model.eval()
    with torch.no_grad():
        #'''
        for (batch_idx, train_tensors) in enumerate(train_loader_eval):
            x = train_tensors[0].to(device)
            labels = train_tensors[1].to(device).float()
            out = model(x).squeeze(-1).squeeze(-1)
            predictions = torch.sigmoid(out)
            predicted_classes = (predictions >= 0.5).float()
            predicted_classes = predicted_classes.squeeze(-1)
            # Compare predicted classes with true labels
            correct = (predicted_classes == labels).float()
            accuracy = correct.mean()
            print(f"Train Accuracy: {accuracy.item() * 100:.2f}%")
        #'''
        for (batch_idx, test_tensors) in enumerate(test_loader):
            x = test_tensors[0].to(device)
            labels = test_tensors[1].to(device).float()
            out = model(x).squeeze(-1).squeeze(-1)
            predictions = torch.sigmoid(out)
            predicted_classes = (predictions >= 0.5).float()
            predicted_classes = predicted_classes.squeeze(-1)
            # Compare predicted classes with true labels
            correct = (predicted_classes == labels).float()
            #print(f"labels.sum(): {labels.sum()}")
            #print(f"predicted_class[0]: {predicted_classes[0]} labels[0]: {labels[0]}")
            accuracy = correct.mean()
            print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
            break