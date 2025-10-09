# src/inference.py
import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
from src.cnn import EmotionCNN
from src.face_detector import detect_face
from torchvision import datasets

# Paths
model_dir = "models"
model_path = os.path.join(model_dir, "emotion_model.pth")
test_dir = "data_cropped/test"  # Needed to get class names

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
# Get classes from dataset
dummy_dataset = datasets.ImageFolder(test_dir)
classes = dummy_dataset.classes

model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_emotion(image_path):
    img = cv2.imread(image_path)
    face_crop = detect_face(img)
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_crop)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face_tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]  # Return class name instead of index

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    label = predict_emotion(args.image)
    print("Predicted class:", label)
