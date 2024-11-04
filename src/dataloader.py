from torch.utils.data import DataLoader
from .dataset import FacialLandmarkDataset

class FacialLandmarkDataloader:
    def __init__(self, data_path, visualize, num_features, store_path, preprocess_data, batch_size, shuffle):
        """
        Initializes the dataloader with the specified parameters.

        Args:
            data_path (str): Path to the directory containing original data.
            visualize (bool): Whether to enable visualization during preprocessing.
            num_features (int): Number of features to extract (default is 136 for landmarks).
            store_path (str): Path to the directory containing .pkl files.
            preprocess_data (bool): Whether to preprocess data before loading.
            batch_size (int): Number of samples per batch. Default is 32.
            shuffle (bool): Whether to shuffle the data. Default is True.
        """
        self.data_path = data_path
        self.viz = visualize
        self.num_features = num_features
        self.store_path = store_path
        self.preprocess_data = preprocess_data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_train_dataloader(self):
        """
        Creates and returns a DataLoader instance for the dataset.

        Returns:
            DataLoader: PyTorch DataLoader instance for the dataset.
        """
        # Initialize the dataset
        dataset = FacialLandmarkDataset(self.data_path, 'train', self.viz, self.num_features, self.store_path, self.preprocess_data)
        
        # Create and return the DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_validation_dataloader(self):
        """
        Creates and returns a DataLoader instance for the dataset.

        Returns:
            DataLoader: PyTorch DataLoader instance for the dataset.
        """
        # TODO separate train, validation, and test data
        # Initialize the dataset
        dataset = FacialLandmarkDataset(self.data_path, 'val', self.viz, self.num_features, self.store_path, self.preprocess_data)
        
        # Create and return the DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def get_test_dataloader(self):
        """
        Creates and returns a DataLoader instance for the dataset.

        Returns:
            DataLoader: PyTorch DataLoader instance for the dataset.
        """
        # TODO separate train, validation, and test data
        # Initialize the dataset
        dataset = FacialLandmarkDataset(self.data_path, 'test', self.viz, self.num_features, self.store_path, self.preprocess_data)
        
        # Create and return the DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
