import os
import json
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

CLASSES: list[str] = [
    "Begging", "Drunkenness", "Fight", "Harassment", "Hijack",
    "Knife hazard", "Normal videos", "Pollution", "Property damage",
    "Robbery", "Terrorism"
]

# Hyperparameters
BATCH_SIZE: int = 8
SEQUENCE_LENGTH: int = 10  # Number of frames in each video sequence
WIDTH: int = 320 # Frame width
HEIGHT: int = 320  # Frame height
LEARNING_RATE: float = 1e-4 # Learning rate for the optimizer
EPOCHS: int = 32 # Number of training epochs
NUM_CHANNELS: int = 3  # Number of channels in the input images (3 for RGB)

def get_class_names() -> list[str]:
    """
    Get the list of class names for the activity recognition dataset.

    Args:
        None

    Returns:
        CLASSES (list[str]): List of class names.
    """
    return CLASSES

def get_num_classes() -> int:
    """
    Get the number of classes in the activity recognition dataset.

    Args:
        None
    
    Returns:
        num_of_classes (int): Number of classes.
    """
    num_of_classes: int = len(get_class_names())
    return num_of_classes

def plot_samples(X: list, Y: list, num_samples: int = 5) -> None:
    """
    Plot a few samples from the dataset.

    Params:
        X (list): List of data samples (images/frames).
        Y (list): List of labels corresponding to the samples.
        num_samples (int): Number of samples to display.

    Returns:
        None
    """
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i][0])  # Display the first frame of each sample
        plt.title(get_class_names()[np.argmax(Y[i])])  # Title is the class name
        plt.axis('off')
    plt.show()
    return

def save_predictions(predictions: dict[str, list[float]], filename: str = 'predictions.json') -> None:
    """
    Save model predictions to a JSON file.

    Args:
        predictions (dict[str, list[float]]): Dictionary where keys are filenames and values are lists of probabilities.
        filename (str): Name of the JSON file to save predictions.

    Returns:
        None
    """
    with open(filename, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    return

def label_video_with_prediction(video_path: str, prediction: list[float], output_dir: str) -> None:
    """
    Label a video with the predicted class and save it to the output directory.

    Args:
        video_path (str): Path to the input video.
        prediction (list[float]): List of probabilities for each class.
        output_dir (str): Directory to save the labeled video.

    Returns:
        None
    """
    class_names: list[str] = get_class_names()
    predicted_class: str = class_names[np.argmax(prediction)]
    output_filename: str = f"{predicted_class}_{os.path.basename(video_path)}"
    output_path: str = os.path.join(output_dir, output_filename)

    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add label to the frame
        cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()

def get_model_predictions(model: torch.nn.Module, input_tensor: torch.Tensor) -> dict[str, list[float]]:
    """
    Get predictions from the model for the given input tensor.

    Args:
        model (torch.nn.Module): The trained model to generate predictions.
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).

    Returns:
        dict[str, list[float]]: Dictionary where keys are class names and values are their corresponding probabilities.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)  # Get model output
        probabilities = torch.softmax(output, dim=1).numpy()  # Apply softmax to get probabilities

    predictions: dict[str, list[float]] = dict()
    class_names: list[str] = get_class_names()
    for i, prob in enumerate(probabilities):
        predictions[class_names[np.argmax(prob)]] = prob.tolist()  # Store the probabilities for the predicted class

    return predictions
