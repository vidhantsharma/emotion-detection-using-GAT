import os
import pickle
from utils import Utils
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms

class FacialLandmarkDataset(Dataset):
    def __init__(self, data_path, visualize, num_features, store_path, preprocess_data):
        self.num_features = num_features
        self.emotion_mapping = {
            "anger" : 1,
            "contempt" : 2,
            "disgust" : 3,
            "fear" : 4,
            "happiness" : 5,
            "neutral" : 6,
            "sadness" : 7,
            "surprise" : 8,
        }
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Ensures the image is grayscale
            transforms.ToTensor()    # Converts image to a tensor
        ])
        if(preprocess_data):
            # Initialize Utils class with parsed arguments
            utils = Utils(data_path=self.data_path, visualize=visualize, num_features=num_features, store_path=store_path, emotion_mapping=self.emotion_mapping)
            # Call methods on the utils instance
            utils.process_pipeline()

        self.file_paths = [os.path.join(store_path, f) for f in os.listdir(store_path) if f.endswith('.pkl')] 
        self.file_names = [f for f in os.listdir(store_path) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load data from a .pkl file
        file_path = self.file_paths[idx]
        file_name = self.file_names[idx]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Extract features and label
        features = data['facial_landmarks'].clone().detach().float()
        edge_index = data['edge_index'].clone().detach().to(torch.uint8)
        label = torch.tensor(data['emotion'], dtype=torch.int8)

        # Extract the image from .pkl name
        filename_split = file_name.split("_")
        image_filename = "_".join(filename_split[3:]).replace(".pkl", "")
        image_emotion_int = int(filename_split[2])
        # search this image_filename in the data_path/emotion
        # Get the corresponding emotion string from the mapping
        image_emotion = next((emotion for emotion, number in self.emotion_mapping.items() if number == image_emotion_int), None)

        full_image_path = os.path.join(self.data_path, image_emotion, image_filename)

        # Check if the image file exists
        if os.path.exists(full_image_path):
            image = Image.open(full_image_path)
            image = image.resize((480,640))
            image = self.transform(image)  # Convert image to tensor
        else:
            image = None
        
        return image, features, edge_index, label
