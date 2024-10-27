import dlib
import cv2
import numpy as np

class ExtractFacialFeatures():
    def __init__(self, data_path, visualize) -> None:
        # Load dlib's face detector and the 68-point landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.visualize = visualize
        self.data_path = data_path

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

            # Normalize landmarks for yaw and roll
            normalized_landmarks = self.normalize_landmarks(landmarks_coords)
            # normalized_landmarks = landmarks_coords

            if self.visualize:
                # Visualize the landmarks on the image
                for (x, y) in normalized_landmarks:
                    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)  # Draw landmarks as green dots

                # Draw a rectangle around the detected face
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle in blue

                # Display the image with landmarks
                cv2.imshow("Landmarks", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return normalized_landmarks

    def normalize_landmarks(self, landmarks_coords):
        # Assuming the first 6 landmarks are the corners of the eyes and mouth
        left_eye = landmarks_coords[36:42]
        right_eye = landmarks_coords[42:48]
        mouth = landmarks_coords[48:60]

        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        # Calculate the angle between the eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        yaw = np.arctan2(dy, dx)

        # Calculate rotation matrix to normalize yaw
        rotation_matrix = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                                     [np.sin(-yaw), np.cos(-yaw)]])
        
        # Normalize landmarks by rotating them
        normalized_landmarks = np.dot(landmarks_coords - left_eye_center, rotation_matrix.T) + left_eye_center

        return normalized_landmarks

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

    def process_data(self) -> np.ndarray:
        # take data_path and create a pickle file

        # Extract facial lanmdmarks features
        landmarks = self.extract_and_visualize_landmarks(self.data_path)
        if landmarks is not None:
            input_data = self.prepare_data(landmarks)
            print(f"Max features in this image : {input_data.shape}")
        else:
            input_data = None
            print("No face detected.")
        return input_data
