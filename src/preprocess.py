import os
from tqdm import tqdm
import cv2
from src.face_detector import detect_face

def preprocess_dataset(input_dir, output_dir):
    """Apply face detection to all images in input_dir and save cropped images to output_dir"""
    for label in os.listdir(input_dir):
        label_in = os.path.join(input_dir, label)
        label_out = os.path.join(output_dir, label)
        os.makedirs(label_out, exist_ok=True)
        for file in tqdm(os.listdir(label_in), desc=f"Processing {label}"):
            img_path = os.path.join(label_in, file)
            img = cv2.imread(img_path)
            face = detect_face(img)
            cv2.imwrite(os.path.join(label_out, file), face)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train", type=str, default="data/train")
    parser.add_argument("--output_train", type=str, default="data_cropped/train")
    parser.add_argument("--input_test", type=str, default="data/test")
    parser.add_argument("--output_test", type=str, default="data_cropped/test")

    args = parser.parse_args()

    preprocess_dataset(args.input_train, args.output_train)
    preprocess_dataset(args.input_test, args.output_test)
