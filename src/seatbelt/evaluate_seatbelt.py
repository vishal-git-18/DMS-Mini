import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from seatbelt_cnn import SeatbeltCNN

# Paths
test_dir = "data_cropped/seatbelt/test"
model_dir = "models"
results_dir = "seatbelt_results"
os.makedirs(results_dir, exist_ok=True)
model_path = os.path.join(model_dir, "seatbelt_model.pth")

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SeatbeltCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Seatbelt Confusion Matrix")
cm_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved at {cm_path}")

# Accuracy bar chart
plt.figure(figsize=(4,4))
sns.barplot(x=["Test"], y=[acc], palette="mako")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Seatbelt Detection Accuracy")
plt.text(0, acc + 0.02, f"{acc:.2f}", ha='center', fontweight='bold')
acc_chart_path = os.path.join(results_dir, "accuracy_chart.png")
plt.savefig(acc_chart_path)
plt.close()
print(f"Accuracy chart saved at {acc_chart_path}")
