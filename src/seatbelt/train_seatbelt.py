import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from seatbelt_cnn import SeatbeltCNN

# Paths
train_dir = "data_cropped/seatbelt/train"
test_dir = "data_cropped/seatbelt/test"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = train_dataset.classes
print("Classes:", classes)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeatbeltCNN(num_classes=len(classes)).to(device)

# Loss & optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

train_acc_list = []

for epoch in range(num_epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    train_acc_list.append(acc)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

# Save model
model_path = os.path.join(model_dir, "seatbelt_model.pth")
torch.save(model.state_dict(), model_path)
print("Seatbelt model saved at", model_path)
