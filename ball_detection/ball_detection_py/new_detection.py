import string
import cv2
from cv2 import imshow
import numpy as np
import imutils

WIDTH = 700
HEIGHT = 500

x_medium = 0
y_medium = 0

border_x = 40
border_y = 40

border_color = (255, 0, 0)
ball_border_color = (0, 255, 0)
thikness = 2


def draw_border(border_x, border_y, border_color, thikness, frame, rows, cols):
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


def draw_countour(border_x, border_y, ball_border_color, thikness, frame, rows, cols, cnt):
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

    return x_medium, y_medium


def sorted_countours(color_mask):
    countours, _ = cv2.findContours(
        color_mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    countours = sorted(
        countours,
        key=lambda x: cv2.contourArea(x),
        reverse=True
    )

    return countours


def adjust_values(x_medium, y_medium, border_x, border_y, rows, cols):
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
    return x_medium, y_medium


def get_the_coordinate_values(border_x, border_y, ball_border_color, thikness, draw_countour, adjust_values, frame, rows, cols, countours, objectType: string):
    for cnt in countours:
        x_medium, y_medium = draw_countour(
            border_x, border_y, ball_border_color, thikness, frame, rows, cols, cnt)

        x_medium, y_medium = adjust_values(
            x_medium, y_medium, border_x, border_y, rows, cols)

        print(f'x_medium for = {objectType}', x_medium,
              f', y_medium for = {objectType}', y_medium)

        break


cap = cv2.VideoCapture(0)
while True:
    # Reading the video from the
    # webcam in image frames
    _, frame = cap.read()
    frame = imutils.resize(frame, width=WIDTH, height=HEIGHT)

    # Rows and Cols in frame
    rows, cols, _ = frame.shape

    # Draw border
    draw_border(border_x, border_y, border_color, thikness, frame, rows, cols)

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set range for orange color and
    # define mask
    ORANGE_MIN = np.array([10, 100, 120], dtype=np.uint8)
    ORANGE_MAX = np.array([25, 255, 255], dtype=np.uint8)
    orange_mask = cv2.inRange(hsv_frame, ORANGE_MIN, ORANGE_MAX)

    # Set range for white color and
    # define mask
    WHITE_MIN = np.array([0, 0, 0], dtype=np.uint8)
    WHITE_MAX = np.array([0, 0, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_frame, WHITE_MIN, WHITE_MAX)

    countours = sorted_countours(orange_mask)

    get_the_coordinate_values(
        border_x,
        border_y,
        ball_border_color,
        thikness,
        draw_countour,
        adjust_values,
        frame, rows,
        cols,
        countours,
        "ball"
    )

    countours = sorted_countours(white_mask)

    get_the_coordinate_values(
        border_x,
        border_y,
        ball_border_color,
        thikness,
        draw_countour,
        adjust_values,
        frame, rows,
        cols,
        countours,
        "goalkeeper"
    )

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
