import os
import pickle

class StoreData:
    def __init__(self, num_features, facial_landmarks, emotion, store_path, index=0) -> None:
        self.num_features = num_features
        self.facial_landmarks = facial_landmarks  # (num_features, )
        self.emotion = emotion  # label idx of emotion in int
        self.store_path = store_path  # path to store the processed data
        self.index = index  # unique index for each data point

        # Ensure the folder exists
        os.makedirs(self.store_path, exist_ok=True)

    def store_data(self) -> None:
        # Validate the number of features; skip if not enough
        if len(self.facial_landmarks) < self.num_features:
            print(f"Skipping data point {self.index} - insufficient features")
            return

        print(f"Number of features extracted in this image : {self.num_features}")
        # Trim features if there are more than required
        if len(self.facial_landmarks) > self.num_features:
            facial_landmarks = self.facial_landmarks[:self.num_features]
        else:
            facial_landmarks = self.facial_landmarks

        # Prepare data to store
        data = {
            "facial_landmarks": facial_landmarks,
            "emotion": self.emotion
        }

        # Unique file naming based on index and emotion label
        file_name = f"emotion_data_{self.emotion}_{self.index}.pkl"
        file_path = os.path.join(self.store_path, file_name)

        # Store data in a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Data stored successfully at {file_path}")
