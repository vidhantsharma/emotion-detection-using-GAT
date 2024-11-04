from .extract_facial_features import ExtractFacialFeatures
from .store_data import StoreData

import torch
import os
from tqdm import tqdm

class Utils:
    def __init__(self, data_path, split, visualize, num_features, store_path, emotion_mapping):
        self.data_path = data_path
        self.visualize = visualize
        self.num_features = num_features
        self.store_path = store_path
        self.emotion_mapping = emotion_mapping
        self.split = split

    def process_pipeline(self):
        # Iterate over each emotion subdirectory in data_path
        for emotion_label in os.listdir(self.data_path):
            emotion_split_path = os.path.join(self.data_path, emotion_label, self.split)
            if not os.path.isdir(emotion_split_path):
                continue  # Skip non-directory files

            # Map the emotion label to an integer
            emotion = self.emotion_mapping.get(emotion_label)
            if emotion is None:
                print(f"Skipping unknown emotion label: {emotion_label}")
                continue

            # Get all image files in the emotion directory
            image_files = [file_name for file_name in os.listdir(emotion_split_path) if os.path.isfile(os.path.join(emotion_split_path, file_name))]

            # Use tqdm to create a progress bar
            with tqdm(total=len(image_files), desc=f'Processing {emotion_label}', unit='image') as pbar:
                for _, file_name in enumerate(image_files):
                    file_path = os.path.join(emotion_split_path, file_name)

                    # Extract landmarks from the image
                    landmarks, edge_index = self.extract_facial_features(file_path)

                    # Store these features as input data for training, including the index
                    self.store_feature_data(landmarks, edge_index, emotion, file_name)

                    # Update the progress bar
                    pbar.update(1)

        print("Processing complete.")

    def extract_facial_features(self, file_path):
        extract_facial_features = ExtractFacialFeatures(file_path, self.visualize)
        facial_landmarks, edge_index = extract_facial_features.process_data()
        if facial_landmarks is not None:
            # convert these landmarks to tensor
            facial_landmarks_tensor = torch.tensor(facial_landmarks, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
        else:
            facial_landmarks_tensor = None
            edge_index_tensor = None

        return facial_landmarks_tensor, edge_index_tensor


    def store_feature_data(self, facial_landmarks, edge_index, emotion, filename):
        store_data = StoreData(self.num_features, facial_landmarks, edge_index, emotion, self.store_path, self.split, filename)
        store_data.store_data()
