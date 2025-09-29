import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Set the random state
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Keep log
plain_train_losses, plain_train_accs = [], []
plain_val_losses, plain_val_accs   = [], []

lrn_train_losses, lrn_train_accs   = [], []
lrn_val_losses, lrn_val_accs     = [], []

# Check the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Define transforms
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2023, 0.1994, 0.2010)

train_tfms = tv.transforms.Compose([
    tv.transforms.RandomCrop(32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean, std),
])

test_tfms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean, std),
])

# Split the dataset
train_dataset = CIFAR10(
    root='data',
    train=True,
    transform=train_tfms,
    download=True
)

val_dataset = CIFAR10(
    root='data',
    train=False,
    transform=test_tfms,
    download=True
)

# Load the data
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False
)

# Display the shape of the dataset
for X, y in val_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Define the network for Local response normalization
class LRN(nn.Module):
    def __init__(self, local_size=5, alpha=1e-4, beta=0.75, k=2):
        super().__init__()
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        # Take the squares
        squared = x.pow(2)

        # padding along channel (e.g: if n=5 then 2 left, 2 right)
        padding = (self.local_size - 1) // 2
        squared_avg = torch.nn.functional.avg_pool3d(
            squared.unsqueeze(1),  # [N, 1, C, H, W]
            kernel_size=(self.local_size, 1, 1),
            stride=1,
            padding=(padding, 0, 0)
        ).squeeze(1)

        # normalization
        s = squared_avg * self.local_size
        denom = (self.k + self.alpha * s).pow(self.beta)
        return x / denom

# Define a normal cnn
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Flatten(),

            nn.Dropout(p=0.1),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, X):
        return self.network(X)

# Define the cnn with lrn
class CNNWithLRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            LRN(local_size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            LRN(local_size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            LRN(local_size=5, alpha=1e-4, beta=0.75, k=2),

            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Flatten(),

            nn.Dropout(p=0.1),

            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, X):
        return self.network(X)
    
plain_cnn = CustomCNN().to(device=device)
cnn_with_lrn = CNNWithLRN().to(device=device)

# Define train and test functions
def train(dataloader, model, loss_fn, optimizer, losses, accs):
    size = len(dataloader.dataset)
    model.train()
    correct, total_loss = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).sum().item()

    avg_loss = total_loss / size
    accuracy = correct / size
    losses.append(avg_loss)
    accs.append(accuracy)

    print(f"Train Acc: {accuracy*100:.2f}% | Train Loss: {avg_loss:.4f}")

def test(dataloader, model, loss_fn, losses, accs):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = test_loss / num_batches
    accuracy = correct / size
    losses.append(avg_loss)
    accs.append(accuracy)

    print(f"Val Acc: {accuracy*100:.2f}% | Val Loss: {avg_loss:.4f}")

# Define loss function, optimizers
EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()

opt_plain = optim.SGD(plain_cnn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
opt_lrn = optim.SGD(cnn_with_lrn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Train the models
for t in range(EPOCHS):
    print(f"[PLAIN] Epoch {t+1}")
    train(train_dataloader, plain_cnn, loss_fn, opt_plain, plain_train_losses, plain_train_accs)
    test(val_dataloader, plain_cnn, loss_fn, plain_val_losses, plain_val_accs)
print("Done!")

for t in range(EPOCHS):
    print(f"[LRN] Epoch {t+1}")
    train(train_dataloader, cnn_with_lrn, loss_fn, opt_lrn, lrn_train_losses, lrn_train_accs)
    test(val_dataloader, cnn_with_lrn, loss_fn, lrn_val_losses, lrn_val_accs)
print("Done!")

# Visualize the metrics
epochs = range(1, EPOCHS+1)
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(epochs, plain_train_losses, label='Plain Train')
plt.plot(epochs, plain_val_losses, label='Plain Val')
plt.plot(epochs, lrn_train_losses, label='LRN Train')
plt.plot(epochs, lrn_val_losses, label='LRN Val')
plt.title('Loss per Epoch') 
plt.legend()

# Acc
plt.subplot(1,2,2)
plt.plot(epochs, [a*100 for a in plain_train_accs], label='Plain Train')
plt.plot(epochs, [a*100 for a in plain_val_accs], label='Plain Val')
plt.plot(epochs, [a*100 for a in lrn_train_accs], label='LRN Train')
plt.plot(epochs, [a*100 for a in lrn_val_accs], label='LRN Val')
plt.title('Accuracy per Epoch (%)')
plt.legend()

plt.savefig("plain_vs_lrn_loss_acc.png", dpi=300, bbox_inches='tight')
plt.show()

# Save the models
torch.save(plain_cnn.state_dict(), 'plain_cnn.pth')
print('Saved plain cnn model to plain_cnn.pth')

torch.save(cnn_with_lrn.state_dict(), 'cnn_with_lrn.pth')
print('Saved cnn with lrn model to cnn_with_lrn.pth')