import cv2
import time
import mediapipe as mp

data_path = input("Enter data path")
print()
frame_rate = input("Enter fps:")


prev = 0
cam = cv2.VideoCapture(0)
while True:
    time_elapsed = time.time() - prev
    success, img = cam.read()

    if time_elapsed > 1.0 / frame_rate:
        prev = time.time()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break