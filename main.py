from pathlib import Path
import cv2 as cv
from utils import run_inference_on_video

# init the webcam
cap = cv.VideoCapture(0)


cwd = Path(__file__).parent.resolve()
model_path = cwd / "hand_gesture_model.pth"
classes_filename = cwd / "classes.txt"

run_inference_on_video(cap=cap, model_path=model_path, class_names_file=classes_filename, confidence_threshold=0.7)

cap.release()
cv.destroyAllWindows()