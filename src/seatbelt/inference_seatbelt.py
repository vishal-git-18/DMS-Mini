import os
import cv2
import torch
from torchvision import transforms
from seatbelt_cnn import SeatbeltCNN

# Paths
model_dir = "models"
model_path = os.path.join(model_dir, "seatbelt_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SeatbeltCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_seatbelt(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image"
    img = cv2.resize(img, (48,48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
    return "Seatbelt" if pred.item()==0 else "No Seatbelt"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    label = predict_seatbelt(args.image)
    print("Prediction:", label)
