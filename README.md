# Hand Gesture Recognition

This repository contains an implementation of a hand gesture recognition system using Mediapipe for hand landmark extraction and a neural network for classification. The model is trained on extracted hand features and can be tested in real-time using a webcam.

## Files in this Repository

- `main.py` - Runs real-time gesture recognition using the user's webcam.
- `utils.py` - Includes all required functions for creating a dataset, hand landmark feature extraction, processing data, training the NN model, and running inferences.
- `classes.txt` - Contains the names of the hand gesture classes.
- `trained_model.pth` - The trained model file for gesture recognition.

## Installation

To set up and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Real-Time Gesture Recognition

To run the model using your webcam, execute the following command:
```bash
python main.py
```

## Feature Extraction

We use Mediapipe's hand-tracking module to extract hand landmarks. Then, we extracted some features from the hand landmarks to detect the hand gestures. The features consist of:
* distances between key landmarks
* angles between fingers
* normalized finger lengths
* overall orientation of the hand

## Model Architecture

The neural network is a multi-layer perceptron (MLP) that takes extracted features from Mediapipe hand landmarks and classifies them into different gesture categories. The model achieved an accuracy of 0.97. Here is the learning curve of the model.

![learning_curves_V4_10_classes](https://github.com/user-attachments/assets/d0d3c746-9be8-43ce-9ba8-a306f25c60b0)

## License
This project is open-source and available under the MIT License.
