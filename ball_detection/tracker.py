import cv2
import numpy as np

ORANGE_MIN = np.array([10, 100, 120], np.uint8)
ORANGE_MAX = np.array([25, 255, 255], np.uint8)

BORDER_COLOR = (255, 0, 0)
BALL_COLOR = (0, 255, 0)
THICKNESS = 2


def _clamp(value, low, high):
    return max(low, min(value, high))


def track_ball(frame, border_x=5, border_y=20, color_min=ORANGE_MIN, color_max=ORANGE_MAX, draw=True):
    """Locate the largest color-matching blob in `frame` within a bordered play area.

    Returns ((x, y), frame): (x, y) is the ball's position relative to the
    play area (clamped to it), or (None, frame) if no blob was found.
    """
    rows, cols, _ = frame.shape
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, color_min, color_max)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rows -= border_y
    cols -= border_x

    if draw:
        cv2.line(frame, (border_x, border_y), (border_x, rows), BORDER_COLOR, THICKNESS)
        cv2.line(frame, (border_x, border_y), (cols, border_y), BORDER_COLOR, THICKNESS)
        cv2.line(frame, (cols, border_y), (cols, rows), BORDER_COLOR, THICKNESS)
        cv2.line(frame, (border_x, rows), (cols, rows), BORDER_COLOR, THICKNESS)

    if not contours:
        return None, frame

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    x_medium = x + w // 2
    y_medium = y + h // 2

    if draw and border_x <= x_medium <= cols and border_y <= y_medium <= rows:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BALL_COLOR, THICKNESS)
        cv2.line(frame, (x_medium, border_y), (x_medium, rows), BALL_COLOR, THICKNESS)
        cv2.line(frame, (border_x, y_medium), (cols, y_medium), BALL_COLOR, THICKNESS)

    x_medium = _clamp(x_medium - border_x, 0, cols - border_x)
    y_medium = _clamp(y_medium - border_y, 0, rows - border_y)

    return (x_medium, y_medium), frame
