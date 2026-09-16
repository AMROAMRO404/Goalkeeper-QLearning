import sys
from pathlib import Path

import cv2
import numpy as np

from arduino import Arduino
from environment import Pong
from helpers import DEFAULT_BAUD, DEFAULT_QTABLE, get_discrete_state, load_q_table
from train import TABLE_SIZE

sys.path.append(str(Path(__file__).resolve().parents[1] / "ball_detection"))
from tracker import track_ball  # noqa: E402


def play(filename=DEFAULT_QTABLE, ball_speed=None, paddle_speed=None,
         port=None, baud=DEFAULT_BAUD, use_arduino=True):
    q_table = load_q_table(filename)
    if q_table is None:
        print('q-table not found:', filename)
        return

    arduino = Arduino(port, baud) if use_arduino else None

    env = Pong(ball_speed=ball_speed, paddle_speed=paddle_speed)
    state = env.reset()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ball_pos, frame = track_ball(frame)
            if ball_pos is not None:
                # only the ball's y-position feeds the trained agent
                _, y_medium = ball_pos
                state[1] = y_medium

            cv2.imshow("Frame", frame)

            discrete_state = get_discrete_state(state, TABLE_SIZE)
            action = np.argmax(q_table[discrete_state])
            if arduino is not None:
                arduino.send_action(action)
            state, _, _ = env.step(action)
            env.render()

            if cv2.waitKey(1) == 27:
                break
    finally:
        if arduino is not None:
            arduino.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    play()
