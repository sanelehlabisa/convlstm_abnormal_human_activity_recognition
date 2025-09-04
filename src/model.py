import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_model_predictions, save_predictions

class ConvLSTMModel(nn.Module):
    def __init__(self, num_classes: int = 11) -> None:
        super(ConvLSTMModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), padding=1)
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=256, batch_first=True)  # Adjust input size based on frame size
        self.fc1 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, C, H, W = x.size()

        # Reshape x for convolutional layers
        x = x.view(batch_size * seq_len, C, H, W)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Downsample

        # Reshape back to (batch_size, seq_len, -1) for LSTM
        x = x.view(batch_size, seq_len, -1)

        # LSTM layer
        x, _ = self.lstm(x)

        # Take the output from the last time step
        x = x[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Fully connected layer
        x = self.dropout(x)
        x = self.fc1(x)

        return x

def main() -> None:
    # Hyperparameters
    batch_size = 8
    sequence_length = 10  # Number of frames in each video sequence
    height, width = 320, 320  # Frame dimensions
    
    # Initialize the model
    model = ConvLSTMModel()
    
    # Create a random input tensor for testing
    input_tensor = torch.randn(batch_size, sequence_length, 3, height, width)

    # Get predictions from the model
    predictions = get_model_predictions(model, input_tensor)

    # Save predictions to JSON
    save_predictions(predictions)

    # Optionally, you can also label videos here if you have video paths and their corresponding predictions
    # for video_name, probs in predictions.items():
    #     label_video_with_prediction(f'path/to/videos/{video_name}', probs, 'predictions/')

if __name__ == "__main__":
    main()

