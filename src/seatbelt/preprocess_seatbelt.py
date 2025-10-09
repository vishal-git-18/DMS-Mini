import os
from tqdm import tqdm
import cv2

def preprocess_dataset(input_dir, output_dir, img_size=(48, 48)):
    """Crop/resize seatbelt images and save to output_dir"""
    for label in os.listdir(input_dir):
        label_in = os.path.join(input_dir, label)
        label_out = os.path.join(output_dir, label)
        os.makedirs(label_out, exist_ok=True)

        for file in tqdm(os.listdir(label_in), desc=f"Processing {label}"):
            img_path = os.path.join(label_in, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Optional: crop ROI if needed (here we just resize)
            img_resized = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(label_out, file), img_resized)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_train", default="data/seatbelt/train")
    parser.add_argument("--output_train", default="data_cropped/seatbelt/train")
    parser.add_argument("--input_test", default="data/seatbelt/test")
    parser.add_argument("--output_test", default="data_cropped/seatbelt/test")
    args = parser.parse_args()

    preprocess_dataset(args.input_train, args.output_train)
    preprocess_dataset(args.input_test, args.output_test)
