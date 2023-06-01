import argparse

from zzsn.train import run_train


class FunctionMapper:
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
    fm = FunctionMapper()

    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="command", required=True)

    train = modes.add_parser("train")
    train.add_argument("--epochs", "-e", type=int, default=5)
    train.add_argument("--n_way", "-nw", type=int, default=5)
    train.add_argument("--n_support", "-ns", type=int, default=5)
    train.add_argument("--n_query", "-nq", type=int, default=5)
    train.add_argument("--n_train_episodes", "-nte", type=int, default=100)
    train.add_argument("--n_eval_episodes", "-nee", type=int, default=10)
    train.add_argument("--learning_rate", "-lr", type=float, default=0.001)
    train.add_argument(
        "--distance_func",
        "-dist",
        type=str,
        choices=["euclidean"],
        default="euclidean",
    )
    train.set_defaults(func=fm.train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
