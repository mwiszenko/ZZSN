import argparse
import subprocess

from zzsn.constants import (
    DEFAULT_DISTANCE_FUNC,
    DEFAULT_DOWNLOAD_DATA,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_EVAL_EPISODES,
    DEFAULT_N_QUERY,
    DEFAULT_N_SUPPORT,
    DEFAULT_N_TRAIN_EPISODES,
    DEFAULT_N_WAY,
    DEFAULT_MODEL,
    DISTANCE_FUNCTIONS,
    OMNIGLOT_SCRIPT_PATH,
    MINIIMAGENET_SCRIPT_PATH,
    DEFAULT_DATASET,
    OMNIGLOT,
    MINIIMAGENET,
    DATASETS,
)
from zzsn.train import run_train
from zzsn.test import run_test


class ModeMapper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def train(args):
        if args.download_data:
            if args.dataset == OMNIGLOT:
                subprocess.call(["sh", OMNIGLOT_SCRIPT_PATH])
            elif args.dataset == MINIIMAGENET:
                subprocess.call(["sh", MINIIMAGENET_SCRIPT_PATH])
        run_train(
            epochs=args.epochs,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            n_train_episodes=args.n_train_episodes,
            n_eval_episodes=args.n_eval_episodes,
            learning_rate=args.learning_rate,
            distance_func=args.distance_func,
            dataset=args.dataset,
        )

    @staticmethod
    def test(args):
        model = args.model[:-4] if args.model[-4:] == ".bin" else args.model
        run_test(
            model=model,
            dataset=args.dataset,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            n_eval_episodes=args.n_eval_episodes,
            distance_func=args.distance_func,
        )

    @staticmethod
    def predict(args):
        pass
        # model, img


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="command", required=True)

    # train
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
    train_mode.add_argument(
        "--download_data",
        "-dd",
        type=bool,
        default=DEFAULT_DOWNLOAD_DATA,
    )
    train_mode.add_argument(
        "--dataset",
        "-ds",
        type=str,
        choices=DATASETS,
        default=DEFAULT_DATASET,
    )
    train_mode.set_defaults(func=ModeMapper.train)

    # test
    test_mode = modes.add_parser("test")
    test_mode.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
    )
    test_mode.add_argument(
        "--dataset",
        "-ds",
        type=str,
        choices=DATASETS,
        default=DEFAULT_DATASET,
    )
    test_mode.add_argument(
        "--n_way", 
        "-nw", 
        type=int, 
        default=DEFAULT_N_WAY
    )
    test_mode.add_argument(
        "--n_support", 
        "-ns", 
        type=int, 
        default=DEFAULT_N_SUPPORT
    )
    test_mode.add_argument(
        "--n_query", 
        "-nq", 
        type=int, 
        default=DEFAULT_N_QUERY
    )
    test_mode.add_argument(
        "--n_eval_episodes", 
        "-nee", 
        type=int, 
        default=DEFAULT_N_EVAL_EPISODES
    )
    test_mode.add_argument(
        "--distance_func",
        "-dist",
        type=str,
        choices=DISTANCE_FUNCTIONS,
        default=DEFAULT_DISTANCE_FUNC,
    )
    test_mode.set_defaults(func=ModeMapper.test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
