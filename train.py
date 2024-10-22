from utils import Utils
import argparse

# Define a function to handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Utility script with visualization options')
    parser.add_argument('--data_path', type=str, required=False, default=r"data")
    parser.add_argument('--viz', action='store_true', help='Enable visualization of the process')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize Utils class with parsed arguments
    utils = Utils(data_path=args.data_path, visualize=args.viz)
    
    # Call methods on the utils instance
    utils.extract_facial_features()