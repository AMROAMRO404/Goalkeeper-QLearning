import cv2
from cv2 import imshow
import numpy as np


x_medium = 0
y_medium = 0

border_x = 5
border_y = 20

border_color = (255, 0, 0)
ball_border_color = (0, 255, 0)

thikness = 2
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    rows, cols, _ = frame.shape

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ORANGE_MIN = np.array([10, 100, 120], np.uint8)
    ORANGE_MAX = np.array([25, 255, 255], np.uint8)

    mask = cv2.inRange(hsv_frame, ORANGE_MIN, ORANGE_MAX)

    countours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    countours = sorted(
        countours,
        key=lambda x: cv2.contourArea(x),
        reverse=True
    )

    rows = rows - border_y
    cols = cols - border_x

    # draw rectangle
    cv2.line(frame,  (border_x, border_y), (border_x, rows),
             border_color, thikness)

    cv2.line(frame,  (border_x, border_y), (cols, border_y),
             border_color, thikness)

    cv2.line(frame,  (cols, border_y), (cols, rows),
             border_color, thikness)

    cv2.line(frame,  (border_x, rows), (cols, rows),
             border_color, thikness)

    for cnt in countours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        x_medium = (x + w//2)
        y_medium = (y + h//2)

        if (x_medium >= border_x and x_medium <= cols) and (y_medium >= border_y and y_medium <= rows):
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          ball_border_color, thikness)
            cv2.line(frame,  (x_medium, border_y),
                     (x_medium, rows), ball_border_color, thikness)
            cv2.line(frame, (border_x, y_medium),
                     (cols, y_medium), ball_border_color, thikness)

        # the values that goes to RL model
        x_medium = x_medium - border_x
        y_medium = y_medium - border_y

        if (x_medium < border_x):
            x_medium = 0

        if (y_medium < border_y):
            y_medium = 0

        if (x_medium > cols - border_x):
            x_medium = cols - border_x

        if (y_medium > rows - border_y):
            y_medium = rows - border_y

        print("x_medium = ", x_medium, ", y_medium = ", y_medium)

        break

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
