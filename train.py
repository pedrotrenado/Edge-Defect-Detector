import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim



# Download and load CIFAR-10 
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# Filter only automobiles (1) and trucks (9)
def filter_classes(dataset, class_a, class_b):
    indices = [i for i, (_, label) in enumerate(dataset)
               if label == class_a or label == class_b]
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

train_subset = filter_classes(dataset, class_a=1, class_b=9)
print("Filtered images:", len(train_subset))

# Remap labels: automobile(1) → 0, truck(9) → 1
class RemapLabels(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        new_label = 0 if label == 1 else 1
        return image, new_label

remapped = RemapLabels(train_subset)

# Split 80/20
train_size = int(0.8 * len(remapped))
val_size = len(remapped) - train_size
train_data, val_data = random_split(remapped, [train_size, val_size])

# DataLoaders — feed images in batches to the network
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

print(f"Train: {train_size} images | Val: {val_size} images")

class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 32x32x3 → 32x32x16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 32x32x16 → 16x16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 16x16x16 → 16x16x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16x32 → 8x8x32
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),          # 8x8x32 = 2048 values
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)      # 2 output classes: good / defect
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = DefectCNN()
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 10

train_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()          # reset gradients
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)  # calculate error
        loss.backward()                # backpropagation
        optimizer.step()               # update weights
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.3f} | Val Accuracy: {acc:.1f}%")
    train_losses.append(running_loss / len(train_loader))
    val_accuracies.append(acc)  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, color='tomato', marker='o')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(val_accuracies, color='steelblue', marker='o')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("Plot saved -> training_results.png")

# Save the trained model
torch.save(model.state_dict(), 'defect_cnn.pth')
print("\nModel saved -> defect_cnn.pth")

# Calculate and print model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {total_params:,} parameters")
print(f"Approx size on disk: {total_params * 4 / 1024:.1f} KB")