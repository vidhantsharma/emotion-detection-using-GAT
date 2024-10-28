from utils import Utils
import argparse
from src.dataloader import FacialLandmarkDataloader

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Utility script with visualization options')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    parser.add_argument('--num_features', type=int, required=False, default=136)
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

    for features, labels in train_loader:
        print(labels)