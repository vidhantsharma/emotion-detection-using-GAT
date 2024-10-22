from .extract_facial_features import ExtractFacialFeatures

class Utils:
    def __init__(self, data_path=None, visualize=False):
        self.data_path = data_path
        self.visualize = visualize

    # Dynamically add the utility functions from other modules
    def extract_facial_features(self):
        extractFacialFeatures = ExtractFacialFeatures(self.data_path, self.visualize)
        extractFacialFeatures.process_data()
