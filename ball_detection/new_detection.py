import cv2

from tracker import track_ball

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    ball_pos, frame = track_ball(frame)
    if ball_pos is not None:
        x_medium, y_medium = ball_pos
        print("x_medium = ", x_medium, ", y_medium = ", y_medium)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
