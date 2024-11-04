import os
import pickle

class StoreData:
    def __init__(self, num_features, facial_landmarks, edge_index, emotion, store_path, split, filename) -> None:
        self.num_features = num_features
        self.facial_landmarks = facial_landmarks  # (num_features, )
        self.emotion = emotion  # label idx of emotion in int
        self.store_path = store_path  # path to store the processed data
        self.filename = filename  # unique index for each data point
        self.edge_index = edge_index # edge indexes for facial landmarks
        self.split = split

        self.store_split_path = os.path.join(self.store_path , self.split)
        # Ensure the folder exists
        os.makedirs(self.store_split_path, exist_ok=True)

    def store_data(self) -> None:

        # Unique file naming based on index and emotion label
        file_name = f"emotion_data_{self.emotion}_{self.filename}.pkl"
        file_path = os.path.join(self.store_split_path, file_name)

        # Validate the number of features; skip if not enough
        if len(self.facial_landmarks) < self.num_features:
            print(f"Skipping data point {file_name} - insufficient features")
            return

        # Trim features if there are more than required
        if len(self.facial_landmarks) > self.num_features:
            print(f"Skipping data point {file_name} - extra unknown features")
            return

        # Prepare data to store
        data = {
            "facial_landmarks": self.facial_landmarks,
            "emotion": self.emotion,
            "edge_index": self.edge_index
        }

        # Store data in a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Data stored successfully at {file_path}")
