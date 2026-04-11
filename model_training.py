import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Image preprocessing
# -------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# -------------------------
# Load dataset
# -------------------------

dataset = datasets.ImageFolder("labeled_data_final/", transform=transform)

print("Classes:", dataset.classes)

# -------------------------
# Train / Validation split
# -------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(dataset.classes)

# -------------------------
# Load ResNet18 model
# -------------------------

model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -------------------------
# Loss & optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -------------------------
# Training loop
# -------------------------

epochs = 10

train_losses = []
val_accuracies = []

for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # -------------------------
    # Validation
    # -------------------------

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

# -------------------------
# Save model
# -------------------------

torch.save(model.state_dict(), "tamil_inscription_model.pth")

print("Model saved!")

# -------------------------
# Plot training graph
# -------------------------

plt.plot(train_losses, label="Train Loss")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.legend()
plt.title("Training Results")
plt.show()