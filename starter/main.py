import argparse
from starter.train_model import train_model
from starter.get_score import get_score
import logging


def go(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.choice == "all" or args.choice == "train_model":
        logging.info("Train/Test model procedure started")
        train_model()

    if args.choice == "all" or args.choice == "get_score":
        logging.info("Score check procedure started")
        get_score()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--choice",
        type=str,
        choices=["train_model",
                 "get_score",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    go(main_args)