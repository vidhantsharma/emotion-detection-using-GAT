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

        # Define the standard edge connections based on dlib's 68-point landmarks
        self.edges = [
            # Jawline
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
            # Left Eyebrow
            (17, 18), (18, 19), (19, 20), (20, 21),
            # Right Eyebrow
            (22, 23), (23, 24), (24, 25), (25, 26),
            # Nose Bridge
            (27, 28), (28, 29), (29, 30),
            # Nostrils
            (31, 32), (32, 33), (33, 34), (34, 35),
            # Left Eye
            (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # Closing the loop for left eye
            # Right Eye
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # Closing the loop for right eye
            # Outer Lips
            (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56),
            (56, 57), (57, 58), (58, 59), (59, 48),  # Closing the loop for outer lips
            # Inner Lips
            (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)  # Closing the loop for inner lips
        ]

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
            # Create bidirectional edges for an undirected graph
            edge_index = []
            for start, end in self.edges:
                edge_index.append([start, end])
                edge_index.append([end, start])  # for undirected connections

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

            return normalized_landmarks, edge_index

    def normalize_landmarks(self, landmarks_coords):
        left_eye = landmarks_coords[36:42]
        right_eye = landmarks_coords[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        # Calculate angle between eyes
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        yaw = np.arctan2(dy, dx)

        # Rotate landmarks to normalize yaw
        rotation_matrix = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                                     [np.sin(-yaw), np.cos(-yaw)]])
        # Normalize landmarks by rotating them
        normalized_landmarks = np.dot(landmarks_coords - left_eye_center, rotation_matrix.T) + left_eye_center
        return normalized_landmarks

    def calculate_mean_edge_angle(self, landmarks, edge_index, eye_line_angle):
        # Calculate the angle of each edge and average for each node
        node_angles = np.zeros(landmarks.shape[0])

        for node in range(landmarks.shape[0]):
            angles = []
            for start, end in edge_index:
                if start == node or end == node:
                    # Calculate angle of edge relative to eye line
                    dx = landmarks[end][0] - landmarks[start][0]
                    dy = landmarks[end][1] - landmarks[start][1]
                    edge_angle = np.arctan2(dy, dx)
                    relative_angle = edge_angle - eye_line_angle
                    angles.append(relative_angle)
            node_angles[node] = np.mean(angles) if angles else 0
        return node_angles

    def prepare_data(self, landmark_coords):
        center_x = np.mean(landmark_coords[:, 0])
        center_y = np.mean(landmark_coords[:, 1])
        center_point = np.array([center_x, center_y])

        # Calculate relative distances of each landmark to the center point
        relative_positions = landmark_coords - center_point

        # Calculate eye line angle
        left_eye_center = np.mean(landmark_coords[36:42], axis=0)
        right_eye_center = np.mean(landmark_coords[42:48], axis=0)
        eye_line_angle = np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0])

        # Calculate mean edge angles
        mean_edge_angles = self.calculate_mean_edge_angle(landmark_coords, self.edges, eye_line_angle)
        
        # Combine relative positions with mean edge angles as features
        input_data = np.hstack((relative_positions, mean_edge_angles[:, np.newaxis]))

        return input_data

    def process_data(self) -> np.ndarray:
        # take data_path and create a pickle file

        # Extract facial lanmdmarks features
        landmarks, edge_index = self.extract_and_visualize_landmarks(self.data_path)
        if landmarks is not None:
            input_data = self.prepare_data(landmarks)
        else:
            input_data = None
            print("No face detected.")
        return input_data, edge_index
