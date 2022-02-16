import cv2
from cv2 import imshow
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    rows, cols, _ = frame.shape

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_value = np.array([161, 155, 84])
    high_value = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low_value, high_value)

    countours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours = sorted(
        countours, key=lambda x: cv2.contourArea(x), reverse=True)

    x_medium = 0
    y_medium = 0
    for cnt in countours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_medium = x + int(w/2)
        y_medium = y + int(h/2)
        print("x_medium = ", x_medium, ", y_medium = ", y_medium)
        cv2.line(frame, (x_medium, 0),
                 (x_medium, rows), (0, 255, 0), 2)
        cv2.line(frame, (0, y_medium),
                 (cols, y_medium), (0, 255, 0), 2)
        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
