import os
import pickle
from utils import Utils
from torch.utils.data import Dataset
import torch

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_path, visualize, num_features, store_path, preprocess_data):
        self.num_features = num_features
        if(preprocess_data):
            # Initialize Utils class with parsed arguments
            utils = Utils(data_path=data_path, visualize=visualize, num_features=num_features, store_path=store_path)
            # Call methods on the utils instance
            utils.process_pipeline()

        self.file_paths = [os.path.join(store_path, f) for f in os.listdir(store_path) if f.endswith('.pkl')] 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load data from a .pkl file
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Extract features and label
        features = torch.tensor(data['facial_landmarks'], dtype=torch.float32)
        label = torch.tensor(data['emotion'], dtype=torch.int8)
        
        return features, label
