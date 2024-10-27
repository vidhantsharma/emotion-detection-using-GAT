from utils import Utils
import argparse

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Utility script with visualization options')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    parser.add_argument('--num_features', type=int, required=False, default=136)
    parser.add_argument('--store_path', type=str, required=False, default=r"processed_data")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize Utils class with parsed arguments
    utils = Utils(data_path=args.data_path, visualize=args.viz, num_features=args.num_features, store_path=args.store_path)
    
    # Call methods on the utils instance
    utils.process_pipeline()