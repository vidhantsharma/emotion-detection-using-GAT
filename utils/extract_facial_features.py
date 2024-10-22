import dlib
import cv2
import numpy as np

class ExtractFacialFeatures():
    def __init__(self, data_path = None, visualize = False) -> None:
        # Load dlib's face detector and the 68-point landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.visualize = visualize
        self.data_path = data_path

    # Function to extract landmarks and visualize them
    def extract_and_visualize_landmarks(self, image_path) -> np.array:
        # Load the image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray)

        # If no faces are detected, return None
        if len(faces) == 0:
            print("No faces detected.")
            return None

        # Extract landmarks for the first detected face and visualize them
        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks_coords = np.array([(p.x, p.y) for p in landmarks.parts()])

            if(self.visualize):
                # Visualize the landmarks on the image
                for (x, y) in landmarks_coords:
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)  # Draw landmarks as green dots

                # Draw a rectangle around the detected face
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle in blue

                # Display the image with landmarks
                cv2.imshow("Landmarks", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return landmarks_coords
        
    def prepare_data(self, landmark_coords):
        center_x = np.mean(landmark_coords[:, 0])
        center_y = np.mean(landmark_coords[:, 1])
        center_point = np.array([center_x, center_y])

        # Calculate relative distances of each landmark to the center point
        relative_positions = landmark_coords - center_point

        # Calculate distances (optional, can be used as features)
        distances = np.linalg.norm(relative_positions, axis=1)

        # Create input data: concatenate relative positions and distances
        input_data = np.concatenate((relative_positions.flatten(), distances))

        return input_data

    def process_data(self):
        # take data_path and create a pickle file
        # Example of extracting landmarks from an image
        image_path = r"data\anger\S014_003_00000030.png"  # Use raw string for the path
        landmarks = self.extract_and_visualize_landmarks(image_path)

        if landmarks is not None:
            input_data = self.prepare_data(landmarks)
            print(f"input feature vector shape : {input_data.shape}")
        else:
            print("No face detected.")
