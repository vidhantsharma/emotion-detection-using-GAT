from utils import Utils
import argparse
from src.dataloader import FacialLandmarkDataloader
from src.model import EmotionClassifier
from src.earlystopping import EarlyStopping

import torch.nn as nn
import torch

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Utility script with visualization options')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    parser.add_argument('--num_nodes', type=int, required=False, default=68, help="number of landmarks")
    parser.add_argument('--num_classes', type=int, required=False, default=8, help="number of emotions")
    parser.add_argument('--store_path', type=str, required=False, default=r"processed_data")
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--preprocess_data', type=bool, required=False, default=False, help="keep true if you want to preprocess the data")
    parser.add_argument('--shuffle', type=bool, required=False, default=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Initialize the custom dataloader
    dataloader = FacialLandmarkDataloader(data_path=args.data_path, visualize=args.viz, 
                                          num_features=args.num_nodes, store_path=args.store_path, 
                                          preprocess_data=args.preprocess_data, batch_size=args.batch_size, 
                                          shuffle=args.shuffle)
    train_loader = dataloader.get_train_dataloader()

    val_loader = dataloader.get_validation_dataloader()

    # Initialize the model
    model = EmotionClassifier(num_classes=args.num_classes, num_landmarks=args.num_nodes)

    # Check if GPU is available and move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.train()  # Set the model to training mode

    criterion = nn.CrossEntropyLoss()  # Define the loss criterion
    # You may want to define an optimizer here as well
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=5, verbose=True)  # Set patience as needed

    try:
        for epoch in range(10):  # Run for a number of epochs
            # Training phase
            model.train()
            for images, features, edge_index, labels in train_loader:
                # Move data to GPU
                images = images.float().to(device)  # Ensure images are float tensors and move to GPU
                features = features.float().to(device)  # Move features to GPU
                edge_index = edge_index.to(device)  # Move edge_index to GPU
                labelsZeroIndex = labels - 1 # sub 1 to make it 0-indexed
                labelsZeroIndex = labelsZeroIndex.long().to(device) # Ensure labels are long tensors

                # Forward pass
                outputs = model(images, features, edge_index)
                
                # Compute loss
                loss = criterion(outputs, labelsZeroIndex)
                
                # Backward pass and optimization step
                model.zero_grad()  # Clear previous gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update the model parameters

                print(f'Epoch: {epoch}, Loss: {loss.item()}')  # Print loss for the epoch

            # Validation phase
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for images, features, edge_index, labels in val_loader:
                    images = images.float().to(device)
                    features = features.float().to(device)
                    edge_index = edge_index.to(device)
                    labelsZeroIndex = labels - 1 # sub 1 to make it 0-indexed
                    labelsZeroIndex = labelsZeroIndex.long().to(device) # Ensure labels are long tensors

                    # Forward pass
                    outputs = model(images, features, edge_index)
                    
                    # Compute loss
                    loss = criterion(outputs, labelsZeroIndex)
                    val_loss += loss.item()

            val_loss /= len(val_loader)  # Average validation loss
            print(f'Epoch: {epoch}, Validation Loss: {val_loss}')

            # Check for early stopping
            if early_stopping(val_loss):
                break

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model.state_dict(), "interrupted_model.pt")  # Save the current model state

    # Save the final model state at the end of training
    print("Saving final model...")
    torch.save(model.state_dict(), "final_model.pt")
