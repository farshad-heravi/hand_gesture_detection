import mediapipe as mp
import cv2 as cv
import numpy as np
import os
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import uuid


def crop_roi(frame, desired_size=(320, 320)):
    """
    Crops the Region of Interest (ROI) from the center of the input frame 
    based on the specified desired size. If the frame is smaller than the desired size, 
    it returns a copy of the original frame without cropping.

    Parameters:
    - frame: Input image (numpy array).
    - desired_size: Tuple (width, height) specifying the desired crop size.

    Returns:
    - frame: Original frame with a rectangle drawn around the cropped region (if cropped).
    - frame_roi: Cropped region of the frame.
    """

    # Get the dimensions of the input frame
    frame_height, frame_width, _ = frame.shape
    crop_width, crop_height = desired_size

    # Check if cropping is needed (if frame is larger than desired size)
    if frame_width > crop_width or frame_height > crop_height:
        # Calculate the top-left and bottom-right coordinates of the cropping box
        x1, y1 = (frame_width - crop_width) // 2, (frame_height - crop_height) // 2
        x2, y2 = x1 + crop_width, y1 + crop_height
        
        # Extract the cropped region from the frame
        frame_roi = frame[y1:y2, x1:x2]
        
        # Draw a blue rectangle on the original frame indicating the cropped area
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    else:
        # If no cropping is needed, return a copy of the original frame
        frame_roi = frame.copy()

    return frame, frame_roi

def restore_roi(frame, frame_roi, desired_size):
    """
    Restores the cropped Region of Interest (ROI) back to its original position 
    in the frame. This function places the given ROI at the center of the frame 
    based on the specified desired size.

    Parameters:
    - frame: Input image (numpy array) where the ROI will be restored.
    - frame_roi: Cropped ROI that needs to be placed back in the frame.
    - desired_size: Tuple (width, height) specifying the dimensions of the ROI.

    Returns:
    - frame: Updated frame with the ROI restored to its original position.
    """

    # Get the dimensions of the input frame
    frame_height, frame_width, _ = frame.shape
    crop_width, crop_height = desired_size

    # Calculate the top-left coordinates for placing the ROI
    x1, y1 = (frame_width - crop_width) // 2, (frame_height - crop_height) // 2
    
    # Restore the ROI by placing it back in the original frame
    frame[y1:y1 + crop_height, x1:x1 + crop_width] = frame_roi

    return frame

def extract_keypoints(results):
    """
    Extracts the 3D coordinates (x, y, z) of hand landmarks from the detection results.

    Parameters:
    - results: The output from a hand tracking model (e.g., MediaPipe Hands).
    
    Returns:
    - A list of lists containing (x, y, z) tuples for each detected hand.
      If no hands are detected, returns None.
    """

    # Check if any hand landmarks are detected
    if not results.multi_hand_landmarks:
        return None

    # Extract (x, y, z) coordinates for each landmark in each detected hand
    return [[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark] 
            for hand_landmarks in results.multi_hand_landmarks]

def util_capture_hand_dataset(cap, desired_size, landmarker, dataset_dir, video=False):
    """
    Captures hand gesture datasets using a webcam and saves frames, keypoints, and annotations.
    
    Parameters:
    - cap: OpenCV VideoCapture object for capturing video frames.
    - desired_size: Tuple (width, height) specifying the cropping dimensions for the ROI.
    - landmarker: Hand tracking model for detecting hand landmarks.
    - dataset_dir: Directory path where the dataset will be stored.
    - video: Boolean flag indicating whether to capture a continuous video sequence or single images.

    Controls:
    - Press 'q' to quit the capture loop.
    - Press 'c' to change the current detection category (waits for a key input).
    - Press 'f' to capture an image (or start/stop recording when video mode is enabled) / to shoot an image at the next handlandmark detection.

    Saves:
    - Cropped hand ROI images.
    - Annotated hand ROI images (with drawn landmarks).
    - Corresponding keypoint data as a pickle (.pkl) file.
    """

    detection_class = None  # Stores the current category for labeling
    state = {
        "shoot": False, 
        "recording": False, 
        "start_recording": False, 
        "save_video": False
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror effect
        frame = cv.flip(frame, 1)
        # Convert frame to RGB (required by the hand tracking model)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Crop the region of interest (ROI) around the hand
        frame, frame_roi = crop_roi(frame, desired_size)

        # Process the hand landmarks
        hand_results = landmarker.process(frame_roi)

        if hand_results.multi_hand_landmarks:
            # Extract keypoints from detected hand landmarks
            keypoints = extract_keypoints(hand_results)

            # Save the cropped ROI if a detection category is selected
            if detection_class is not None:
                cat_dir = os.path.join(dataset_dir, detection_class)
                if not os.path.exists(cat_dir):
                    os.mkdir(cat_dir)

                if video:
                    # Start recording if the state is set
                    if state['start_recording']:
                        state["start_recording"], state["recording"] = False, True
                    
                    if state['recording']:
                        # Create an annotated copy of the frame with landmarks drawn
                        frame_roi_annotated = frame_roi.copy()
                        frame_roi_annotated = landmarker.draw_landmarks(frame_roi_annotated, hand_results)

                        # Convert frames back to BGR before saving
                        frame_roi = cv.cvtColor(frame_roi, cv.COLOR_RGB2BGR)
                        frame_roi_annotated = cv.cvtColor(frame_roi_annotated, cv.COLOR_RGB2BGR)

                        # Generate a unique filename
                        filename_core = uuid.uuid1()

                        # Save raw and annotated images
                        cv.imwrite(os.path.join(cat_dir, f"{filename_core}.png"), frame_roi)
                        cv.imwrite(os.path.join(cat_dir, f"{filename_core}_annotated.png"), frame_roi_annotated)

                        # Save keypoints as a pickle file
                        with open(os.path.join(cat_dir, f"{filename_core}.pkl"), 'wb') as file:
                            pickle.dump(keypoints, file)

                    # Stop recording when the "save_video" flag is triggered
                    if state['save_video']:
                        state["recording"], state["save_video"] = False, False

                else:
                    # Save a single frame when "shoot" is triggered
                    if state['shoot']:
                        filename_core = uuid.uuid1()
                        frame_roi = cv.cvtColor(frame_roi, cv.COLOR_RGB2BGR)

                        # Save image and keypoints
                        cv.imwrite(os.path.join(cat_dir, f"{filename_core}.png"), frame_roi)
                        with open(os.path.join(cat_dir, f"{filename_core}.pkl"), 'wb') as file:
                            pickle.dump(keypoints, file)

                        print(f"Saved {filename_core}")
                        state["shoot"] = False

            # Draw landmarks on the ROI and restore it to the original frame
            frame_roi_annotated = landmarker.draw_landmarks(frame_roi, hand_results)
            frame = restore_roi(frame, frame_roi_annotated, desired_size)
        
        # Convert frame back to BGR for OpenCV display
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("image", frame)

        # Handle user input
        key = cv.waitKey(1)
        if key == ord('q'):
            break  # Exit loop
        elif key == ord('c'):
            # Change detection category
            class_no = chr(cv.waitKey(0))
            detection_class = class_no
        elif key == ord('f'):
            # Capture a frame or handle video recording states
            if video:
                if not state['recording']:
                    state['start_recording'] = True  # Start recording
                else:
                    state['save_video'] = True  # Stop recording
            else:
                state['shoot'] = True  # Capture a single frame

def extract_features(results):
    """Extracts meaningful features from Mediapipe hand landmarks, including hand orientation."""

    if not results:
        return np.array([])  # Return empty if no hands detected

    def distance(landmarks, p1, p2):
        """Euclidean distance between two points."""
        p1, p2 = landmarks[p1], landmarks[p2]
        return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])

    def angle_between(p1, p2, p3):
        """Computes the angle (in radians) between three points (p1-p2-p3)."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(dot_product / norm_product) if norm_product else 0.0

    features = []
    lmk = results[0]  # Assuming only one hand detected

    # Distance features (between key landmarks)
    distances = np.array([
        distance(lmk, 4, 8),  # Thumb to Index
        distance(lmk, 4, 12), # Thumb to Middle
        distance(lmk, 4, 16), # Thumb to Ring
        distance(lmk, 4, 20), # Thumb to Pinky
        distance(lmk, 0, 4),  # Wrist to Thumb Tip
        distance(lmk, 0, 8),  # Wrist to Index Tip
        distance(lmk, 0, 12), # Wrist to Middle Tip
        distance(lmk, 0, 16), # Wrist to Ring Tip
        distance(lmk, 0, 20)  # Wrist to Pinky Tip
    ])

    # Angle features (between fingers)
    angles = np.array([
        angle_between(lmk[0], lmk[5], lmk[8]),  # Wrist -> MCP -> Index Tip
        angle_between(lmk[0], lmk[9], lmk[12]), # Wrist -> MCP -> Middle Tip
        angle_between(lmk[0], lmk[13], lmk[16]),# Wrist -> MCP -> Ring Tip
        angle_between(lmk[0], lmk[17], lmk[20]) # Wrist -> MCP -> Pinky Tip
    ])

    # Finger lengths (normalized by palm size)
    palm_size = distance(lmk, 0, 9)  # Wrist to Middle MCP
    finger_lengths = np.array([
        distance(lmk, 0, 8) / palm_size,  # Index
        distance(lmk, 0, 12) / palm_size, # Middle
        distance(lmk, 0, 16) / palm_size, # Ring
        distance(lmk, 0, 20) / palm_size  # Pinky
    ])

    # **Hand Orientation Angle Calculation**
    wrist = np.array(lmk[0])   # Wrist landmark
    middle_finger_base = np.array(lmk[9])  # Middle finger base
    hand_vector = middle_finger_base - wrist
    hand_angle = np.arctan2(hand_vector[1], hand_vector[0]) * 180 / np.pi  # Convert to degrees

    # Combine all features
    emb = np.concatenate((distances, angles, finger_lengths, [hand_angle]))

    # Normalize feature vector
    norm = np.linalg.norm(emb) + 1e-8  # Avoid division by zero
    features.append(emb / norm)

    return np.array(features)

def load_or_process_data(processed_data_path, dataset_path):
    """Loads cached feature data if available; otherwise, processes video files and extracts hand orientation angle."""
    
    if os.path.exists(processed_data_path):
        print("Loading cached dataset...")
        with open(processed_data_path, "rb") as f:
            return pickle.load(f)

    print("Processing videos for feature extraction...")
    X_data, Y_data = [], []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        for pkl_file in Path(category_path).glob('*.pkl'):
            with open(pkl_file, 'rb') as file:
                results = pickle.load(file)

            emb = extract_features(results)  # Uses the updated function with hand orientation

            if emb.sum() != 0:  # Ensure valid features
                X_data.append(emb)
                Y_data.append(int(category))

    # Convert lists to numpy arrays for consistency
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Save processed data
    with open(processed_data_path, "wb") as f:
        pickle.dump((X_data, Y_data), f)

    print("Feature extraction completed and cached.")
    return X_data, Y_data

def train_model(processed_data_path, dataset_path, model_path, batch_size=32, num_class=10):
    """Trains a neural network on extracted hand gesture features, including hand orientation."""

    # Load preprocessed feature dataset
    X, Y = load_or_process_data(processed_data_path=processed_data_path, dataset_path=dataset_path)

    # Convert data to PyTorch tensors
    X_tensor, Y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

    # Shuffle dataset
    indices = torch.randperm(len(X_tensor))
    X_tensor, Y_tensor = X_tensor[indices], Y_tensor[indices]

    # Train-validation split (80%-20%)
    split = int(0.8 * len(X))
    X_train, X_val = X_tensor[:split], X_tensor[split:]
    Y_train, Y_val = Y_tensor[:split], Y_tensor[split:]

    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_tensor.shape[1]  # Ensure correct feature size, including orientation
    model = HandGestureNet(num_classes=num_class, input_size=input_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Training settings
    epochs = 200
    train_losses, val_losses, val_accuracies = [], [], []
    forceStopTraining = False

    # Live Plotting
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for epoch in range(epochs):
        if forceStopTraining:
            break

        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # Adjust learning rate

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                val_outputs = model(X_batch)
                val_outputs = val_outputs.squeeze(1)
                loss = criterion(val_outputs, Y_batch)
                val_loss += loss.item()

                predictions = torch.argmax(val_outputs, dim=1)
                correct += (predictions == Y_batch).sum().item()
                total += Y_batch.size(0)

        # Compute metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f}")

        # Live updating plot
        ax[0].clear()
        ax[0].plot(train_losses, label="Train Loss")
        ax[0].plot(val_losses, label="Validation Loss")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].clear()
        ax[1].plot(val_accuracies, label="Validation Accuracy", color="green")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        plt.pause(0.1)

        if cv.waitKey(1) == ord('q'):  # Press 'q' to stop training
            forceStopTraining = True

    plt.ioff()
    plt.show()

    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model saved at {model_path}.")

def run_inference_on_video(cap, model_path, video_path=None, class_names_file = 'clasees.txt', num_class=10, confidence_threshold=0.7):
    """Runs real-time hand gesture recognition on a video stream."""

    # load classes name
    GESTURE_CLASSES = {}
    with open(class_names_file, 'r') as file:
        for i, row in enumerate(file.readlines()):
            GESTURE_CLASSES[i] = row.removesuffix('\n')

    # Load trained model
    model = HandGestureNet(num_classes=num_class, input_size=18)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize hand landmark detector
    hand_landmark = HandLandmarker(mode='image', max_num_hands=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process frame to detect hand landmarks
        results = hand_landmark.process(frame_rgb)
        hand_landmark.draw_landmarks(frame)

        if results.multi_hand_landmarks:
            keypoints = extract_keypoints(results)

            emb = extract_features(keypoints)
            emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(emb)
                output = torch.softmax(output.squeeze(1), dim=1)  # Convert to probabilities

                predicted_class = torch.argmax(output, dim=1).item()
                confidence = output[0, predicted_class].item()

                if confidence >= confidence_threshold:
                    gesture_name = GESTURE_CLASSES.get(predicted_class, "Unknown Gesture")

                # Display gesture prediction and confidence on screen
                cv.putText(frame, f"{gesture_name} ({confidence:.2f})", (50, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show output frame
        cv.imshow("Hand Gesture Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv.destroyAllWindows()


### classes
class HandLandmarker():
    def __init__(self, mode='image', max_num_hands=1, 
                 min_detection_confidence=0.8,
                 min_tracking_confidence=0.9):
        self.results = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(True if mode == "image" else False,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence)
        
    def process(self, frame):
        self.results = self.hands.process(frame)
        return self.results
    
    def draw_landmarks(self, frame, results=None):
        results = results or self.results
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        return frame
    
    def close(self):
        self.hands.close()

class HandGestureNet(nn.Module):
    def __init__(self, num_classes, input_size=20):
        super(HandGestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x