from utils import Utils
import argparse
from src.dataloader import FacialLandmarkDataloader
from src.model import EmotionClassifier

import torch.nn as nn
import torch

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Utility script with visualization options')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    parser.add_argument('--num_features', type=int, required=False, default=136)
    parser.add_argument('--num_classes', type=int, required=False, default=8)
    parser.add_argument('--num_landmarks', type=int, required=False, default=136)
    parser.add_argument('--store_path', type=str, required=False, default=r"processed_data")
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--preprocess_data', type=bool, required=False, default=False)
    parser.add_argument('--shuffle', type=bool, required=False, default=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Initialize the custom dataloader
    dataloader = FacialLandmarkDataloader(data_path=args.data_path, visualize=args.viz, 
                                          num_features=args.num_features, store_path=args.store_path, 
                                          preprocess_data=args.preprocess_data, batch_size=args.batch_size, 
                                          shuffle=args.shuffle)
    train_loader = dataloader.get_dataloader()

    # Initialize the model
    model = EmotionClassifier(num_classes=args.num_classes, num_landmarks=args.num_landmarks)
    model.train()  # Set the model to training mode

    for epoch in range(10):  # Run for a number of epochs
        for images, features, edge_index, labels in train_loader:
            # Assuming images are in a compatible format (e.g., normalized tensor)
            images = images.float()  # Ensure images are float tensors
            labels = labels.long()  # Ensure labels are long tensors
            
            # Forward pass
            outputs = model(images, features, edge_index)
            
            # Compute loss (you will need to define a criterion, e.g., CrossEntropyLoss)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization step
            model.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            # Here you would typically have an optimizer step, e.g., optimizer.step()

            print(f'Epoch: {epoch}, Loss: {loss.item()}')  # Print loss for the epoch