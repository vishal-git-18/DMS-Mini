import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.cnn import EmotionCNN

# Paths
test_dir = "data_cropped/test"
model_dir = "models"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
model_path = os.path.join(model_dir, "emotion_model.pth")

# Dataset transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test dataset
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
classes = test_dataset.classes

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=len(classes)).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

# Evaluation on test dataset
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Test accuracy
test_acc = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", test_acc)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved at {cm_path}")

# Bar chart: Test Accuracy only
plt.figure(figsize=(4,4))
sns.barplot(x=["Test"], y=[test_acc], palette="viridis")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.text(0, test_acc + 0.02, f"{test_acc:.2f}", ha='center', fontweight='bold')
acc_chart_path = os.path.join(results_dir, "test_accuracy_chart.png")
plt.savefig(acc_chart_path)
plt.close()
print(f"Test accuracy chart saved at {acc_chart_path}")
