import argparse

from helpers import DEFAULT_BALL_SPEED, DEFAULT_BAUD, DEFAULT_PADDLE_SPEED, DEFAULT_QTABLE


def add_speed_args(parser):
    parser.add_argument(
        "--ball-speed", type=int, default=None,
        help=f"ball speed magnitude (default: {DEFAULT_BALL_SPEED})")
    parser.add_argument(
        "--paddle-speed", type=int, default=None,
        help=f"paddle movement speed per step (default: {DEFAULT_PADDLE_SPEED})")


def main():
    parser = argparse.ArgumentParser(description="Goalkeeper Q-learning")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser(
        "train", help="train the Q-learning agent against the simulated ball")
    train_parser.add_argument(
        "--episodes", type=int, default=None,
        help="number of episodes to train for (default: 5001)")
    add_speed_args(train_parser)

    sim_parser = subparsers.add_parser(
        "sim", help="watch a trained agent play against the simulated ball (no webcam)")
    sim_parser.add_argument(
        "--qtable", default=DEFAULT_QTABLE,
        help=f"q-table filename under qtables/ (default: {DEFAULT_QTABLE})")
    sim_parser.add_argument(
        "--episodes", type=int, default=None,
        help="number of episodes to play (default: run forever)")
    add_speed_args(sim_parser)

    play_parser = subparsers.add_parser(
        "play", help="drive the trained agent from a live webcam-tracked ball")
    play_parser.add_argument(
        "--qtable", default=DEFAULT_QTABLE,
        help=f"q-table filename under qtables/ (default: {DEFAULT_QTABLE})")
    play_parser.add_argument(
        "--port", default=None,
        help="Arduino serial port, e.g. /dev/cu.usbmodem14101 or COM3 (default: auto-detect)")
    play_parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD,
        help=f"serial baud rate, must match the sketch (default: {DEFAULT_BAUD})")
    play_parser.add_argument(
        "--no-arduino", action="store_true",
        help="run without sending commands to the Arduino")
    add_speed_args(play_parser)

    human_parser = subparsers.add_parser(
        "human", help="manually control the paddle with the keyboard (W/S)")
    add_speed_args(human_parser)

    args = parser.parse_args()

    if args.mode == "train":
        from train import train
        train(episodes=args.episodes, ball_speed=args.ball_speed, paddle_speed=args.paddle_speed)
    elif args.mode == "sim":
        from sim import sim
        sim(filename=args.qtable, episodes=args.episodes,
            ball_speed=args.ball_speed, paddle_speed=args.paddle_speed)
    elif args.mode == "play":
        from play import play
        play(filename=args.qtable, ball_speed=args.ball_speed, paddle_speed=args.paddle_speed,
             port=args.port, baud=args.baud, use_arduino=not args.no_arduino)
    elif args.mode == "human":
        from human_play import play_human
        play_human(ball_speed=args.ball_speed, paddle_speed=args.paddle_speed)


if __name__ == "__main__":
    main()
