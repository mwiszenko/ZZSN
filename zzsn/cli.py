import argparse

from zzsn.constants import (
    DEFAULT_DISTANCE_FUNC,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_EVAL_EPISODES,
    DEFAULT_N_QUERY,
    DEFAULT_N_SUPPORT,
    DEFAULT_N_TRAIN_EPISODES,
    DEFAULT_N_WAY,
    DISTANCE_FUNCTIONS,
)
from zzsn.train import run_train


class ModeMapper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train(args):
        run_train(
            epochs=args.epochs,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            n_train_episodes=args.n_train_episodes,
            n_eval_episodes=args.n_eval_episodes,
            learning_rate=args.learning_rate,
            distance_func=args.distance_func,
        )


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="command", required=True)

    train_mode = modes.add_parser("train")
    train_mode.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS)
    train_mode.add_argument("--n_way", "-nw", type=int, default=DEFAULT_N_WAY)
    train_mode.add_argument(
        "--n_support", "-ns", type=int, default=DEFAULT_N_SUPPORT
    )
    train_mode.add_argument(
        "--n_query", "-nq", type=int, default=DEFAULT_N_QUERY
    )
    train_mode.add_argument(
        "--n_train_episodes",
        "-nte",
        type=int,
        default=DEFAULT_N_TRAIN_EPISODES,
    )
    train_mode.add_argument(
        "--n_eval_episodes", "-nee", type=int, default=DEFAULT_N_EVAL_EPISODES
    )
    train_mode.add_argument(
        "--learning_rate", "-lr", type=float, default=DEFAULT_LEARNING_RATE
    )
    train_mode.add_argument(
        "--distance_func",
        "-dist",
        type=str,
        choices=DISTANCE_FUNCTIONS,
        default=DEFAULT_DISTANCE_FUNC,
    )
    train_mode.set_defaults(func=ModeMapper.train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
