import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from utils import get_class_names  # Importing the utility function

class ActivityDataset:
    def __init__(self, input_dir, augmentations=None):
        self.input_dir = input_dir
        self.X = []
        self.Y = []
        self.class_counts = {class_name: None for class_name in get_class_names()}
        self.augmentations = augmentations if augmentations else self.default_augmentations()
        self.num_classes = len(self.class_counts)

    def default_augmentations(self):
        # Define default augmentations
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((128, 128)),  # Resize for consistency
            transforms.ToTensor()
        ])

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def apply_augmentations(self, frame):
        pil_image = Image.fromarray(frame)  # Convert to PIL Image for augmentation
        return self.augmentations(pil_image)

    def load_data(self):
        for i, class_name in enumerate(get_class_names()):
            print(f"Processing class: {class_name}")
            y_t = np.zeros(self.num_classes)
            y_t[i] = 1  # One-hot encoding for the class label

            class_dir = os.path.join(self.input_dir, class_name)
            list_files = os.listdir(class_dir)

            self.class_counts[class_name] = len(list_files)  # Count videos in the class

            for file_name in list_files:
                video_path = os.path.join(class_dir, file_name)
                frames = self.extract_frames(video_path)

                # Apply augmentations to each frame
                augmented_frames = [self.apply_augmentations(frame) for frame in frames]
                self.X.append(augmented_frames)
                self.Y.append(y_t)

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def get_train_val_test_split(self, test_size=0.1, val_size=0.2):
        # First, split into train + val and test
        X_temp, X_test, y_temp, y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)

        # Now, split the temp into train and validation
        val_ratio = val_size / (1 - test_size)  # Adjust for the reduced dataset
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

def main() -> None:
    dataset = ActivityDataset(input_dir='path/to/dataset')
    dataset.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset.get_train_val_test_split()
    return

if __name__ == "__main__":
    main()
