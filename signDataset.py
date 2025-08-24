import os
from pathlib import Path
import cv2 as cv
import numpy as np

# Dataset path
data_path = Path("Gestures/")
print(os.getcwd())

# Create base folder if not exists
if not data_path.is_dir():
    data_path.mkdir(parents=True, exist_ok=True)

# Gesture categories
poses = ["up","down","right","left","come","stop","turn","blank"]

# Create folders for each gesture
for pose in poses:
    pose_path = data_path/pose
    pose_path.mkdir(parents=True, exist_ok=True)

# Start camera
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    count = {
        "u": len(os.listdir(data_path/"up")),
        "d": len(os.listdir(data_path/"down")),
        "r": len(os.listdir(data_path/"right")),
        "l": len(os.listdir(data_path/"left")),
        "c": len(os.listdir(data_path/"come")),
        "s": len(os.listdir(data_path/"stop")),
        "t": len(os.listdir(data_path/"turn")),
        "b": len(os.listdir(data_path/"blank")),
    }

    cv.rectangle(frame,(0,40),(300,300),(0,255,0),2)
    roi = frame[40:300, 0:300]
    cv.imshow("ROI", roi)

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    mask = cv.resize(thresh, (256,256))

    cv.imshow("Mask", mask)

    interrupt = cv.waitKey(10)

    if interrupt & 0xFF == ord('u'):
        cv.imwrite(os.path.join("Gestures/up", f"{count['u']}.jpg"), mask)
    if interrupt & 0xFF == ord('d'):
        cv.imwrite(os.path.join("Gestures/down", f"{count['d']}.jpg"), mask)
    if interrupt & 0xFF == ord('r'):
        cv.imwrite(os.path.join("Gestures/right", f"{count['r']}.jpg"), mask)
    if interrupt & 0xFF == ord('l'):
        cv.imwrite(os.path.join("Gestures/left", f"{count['l']}.jpg"), mask)
    if interrupt & 0xFF == ord('c'):
        cv.imwrite(os.path.join("Gestures/come", f"{count['c']}.jpg"), mask)
    if interrupt & 0xFF == ord('s'):
        cv.imwrite(os.path.join("Gestures/stop", f"{count['s']}.jpg"), mask)
    if interrupt & 0xFF == ord('t'):
        cv.imwrite(os.path.join("Gestures/turn", f"{count['t']}.jpg"), mask)
    if interrupt & 0xFF == ord('b'):
        cv.imwrite(os.path.join("Gestures/blank", f"{count['b']}.jpg"), mask)

    if interrupt & 0xFF == ord('k'):
        break

capture.release()
cv.destroyAllWindows()

