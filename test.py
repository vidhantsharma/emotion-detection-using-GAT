import torch
import argparse
from sklearn.metrics import confusion_matrix, f1_score  # Import confusion matrix and F1 score
import numpy as np
from src.dataloader import FacialLandmarkDataloader
from src.model import EmotionClassifier
from tqdm import tqdm

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Testing script for Emotion Classifier')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--model_path', type=str, required=False, default='best_model.pt', help='trained model path')
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    parser.add_argument('--num_nodes', type=int, required=False, default=68)
    parser.add_argument('--num_classes', type=int, required=False, default=8)
    parser.add_argument('--store_path', type=str, required=False, default=r"processed_data")
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--preprocess_data', type=bool, required=False, default=False)
    parser.add_argument('--shuffle', type=bool, required=False, default=True)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Initialize the custom dataloader for testing
    dataloader = FacialLandmarkDataloader(data_path=args.data_path, visualize=args.viz, 
                                          num_features=args.num_nodes, store_path=args.store_path, 
                                          preprocess_data=args.preprocess_data, batch_size=args.batch_size, 
                                          shuffle=args.shuffle)
    test_loader = dataloader.get_test_dataloader()

    # Initialize the model
    model = EmotionClassifier(num_classes=args.num_classes, num_landmarks=args.num_nodes)
    model.load_state_dict(torch.load(args.model_path))  # Load the trained model
    model.eval()  # Set the model to evaluation mode

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for images, features, edge_index, labels in tqdm(test_loader, desc="Testing"):
            images = images.float().to(device)  # Move images to the correct device
            features = features.float().to(device)  # Move features to the correct device
            edge_index = edge_index.to(device)  # Move edge index to the correct device
            labelsZeroIndex = labels - 1  # Subtract 1 to make labels 0-indexed
            labelsZeroIndex = labelsZeroIndex.long().to(device)  # Ensure labels are long tensors

            # Forward pass
            outputs = model(images, features, edge_index)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Store labels and predictions for confusion matrix and F1 score
            all_labels.extend(labelsZeroIndex.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Calculate total correct and samples
            total_correct += (predicted == labelsZeroIndex).sum().item()
            total_samples += labels.size(0)

    # Calculate accuracy
    accuracy = total_correct / total_samples * 100
    print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

    # Calculate confusion matrix and F1 score
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')  # Weighted for class imbalance

    print("Confusion Matrix:")
    print(conf_matrix)
    print(f'F1 Score: {f1:.2f}')
