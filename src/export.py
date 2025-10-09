import os
import torch
from src.cnn import EmotionCNN
from torch import nn
from onnxruntime.quantization import quantize_dynamic, QuantType

model_dir = "models"
onnx_fp32 = os.path.join(model_dir, "emotion_model.onnx")
onnx_int8 = os.path.join(model_dir, "emotion_model_int8.onnx")
model_path = os.path.join(model_dir, "emotion_model.pth")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PyTorch model
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Dummy input
dummy_input = torch.randn(1,1,48,48).to(device)

# Export ONNX FP32
torch.onnx.export(model, dummy_input, onnx_fp32, opset_version=11)
print("ONNX model exported:", onnx_fp32)
print("Size (MB):", os.path.getsize(onnx_fp32)/1e6)

# Quantize to INT8
quantize_dynamic(
    model_input=onnx_fp32,
    model_output=onnx_int8,
    weight_type=QuantType.QInt8
)
print("INT8 quantized ONNX model:", onnx_int8)
print("Size (MB):", os.path.getsize(onnx_int8)/1e6)
