import os
import pickle
from utils import Utils
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from collections import Counter
import random

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_path, split, visualize, num_features, store_path, preprocess_data):
        self.num_features = num_features
        self.emotion_mapping = {
            "anger": 1,
            "disgust": 2,
            "fear": 3,
            "happiness": 4,
            "neutral": 5,
            "surprise": 6,
        }
        self.data_path = data_path
        self.split = split
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Ensures the face is grayscale
        ])
        self.face_detector = MTCNN(image_size=224, margin=20)  # Resize cropped face to 224x224

        # Process pipeline if preprocessing is enabled
        if preprocess_data:
            utils = Utils(data_path=self.data_path, split=self.split, visualize=visualize, 
                          num_features=num_features, store_path=store_path, emotion_mapping=self.emotion_mapping)
            utils.process_pipeline()

        self.file_paths = [os.path.join(store_path, split, f) for f in os.listdir(os.path.join(store_path, split)) if f.endswith('.pkl')] 
        self.file_names = [f for f in os.listdir(os.path.join(store_path, split)) if f.endswith('.pkl')]

        if self.split == 'train':
            # Calculate the class distribution from filenames
            self.class_distribution = Counter([file_name.split('_')[2] for file_name in self.file_names])  # Extract class labels from filenames
            
            # Determine the maximum class frequency
            max_class_freq = max(self.class_distribution.values())

            # Oversample minority classes
            self.oversample_indices = []
            for i, file_name in enumerate(self.file_names):
                class_label = file_name.split('_')[2]
                # Calculate how many times to sample this class based on its frequency
                num_oversamples = max_class_freq // self.class_distribution[class_label]
                for _ in range(num_oversamples):
                    self.oversample_indices.append(i)
        else:
            # For validation or test, use the indices directly
            self.oversample_indices = list(range(len(self.file_names)))

    def __len__(self):
        return len(self.oversample_indices)  # Return the oversampled length

    def __getitem__(self, idx):
        # Get the oversampled index
        file_idx = self.oversample_indices[idx]
        file_path = self.file_paths[file_idx]
        file_name = self.file_names[file_idx]
        
        # Load data from a .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Extract features and label
        features = data['facial_landmarks'].clone().detach().float()
        edge_index = data['edge_index'].clone().detach().to(torch.uint8)
        label = torch.tensor(data['emotion'], dtype=torch.int8)

        # Extract the image file name from the .pkl name
        filename_split = file_name.split("_")
        image_filename = "_".join(filename_split[3:]).replace(".pkl", "")
        image_emotion_int = int(filename_split[2])

        # Get corresponding emotion string from the mapping
        image_emotion = next((emotion for emotion, number in self.emotion_mapping.items() if number == image_emotion_int), None)
        full_image_path = os.path.join(self.data_path, image_emotion, self.split, image_filename)

        # Check if the image file exists and detect the face
        if os.path.exists(full_image_path):
            image = Image.open(full_image_path).convert("RGB")  # Ensure the image is RGB

            # Detect the face and crop it
            face = self.face_detector(image)
            if face is None:
                # If no face is detected, create a zero tensor of the expected size
                face = torch.zeros(1, 224, 224)
            else:
                face = face.squeeze(0)  # Remove batch dimension

            # Apply transformations to the face
            image = self.transform(face)
        else:
            # If image file doesn't exist, create a zero tensor
            image = torch.zeros(1, 224, 224)
        
        return image, features, edge_index, label
