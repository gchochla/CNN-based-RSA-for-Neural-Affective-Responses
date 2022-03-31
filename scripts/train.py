import argparse
import logging

import torch

from project.image_encoders import model_mapping, AVAILABLE_MODELS
from project.predictors import Predictor
from project.datasets import IAPSDataset
from project.trainer import EmbeddingsTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", required=True, type=str, help="tweet dataset directory"
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="train batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="eval batch size",
    )
    parser.add_argument(
        "--model",
        default="densenet",
        choices=AVAILABLE_MODELS,
        type=str,
        help="which resnet to use",
    )
    parser.add_argument(
        "--model_arg",
        type=int,
        help="argument for model initialization, e.g. resnet depth",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="dropout on embeddings"
    )
    parser.add_argument(
        "--crop_size", default=224, type=int, help="dimension to crop images to"
    )
    parser.add_argument(
        "--lr", default=5e-5, type=float, help="learning rate of all models"
    )
    parser.add_argument(
        "--num_train_epochs", default=8, type=int, help="training epochs"
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="ratio of training steps (not epochs)"
        " to warmup lr before linear decay",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="whether to use GPU"
    )
    parser.add_argument(
        "--logging_level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "CRITICAL", "ERROR", "FATAL"],
        help="level of logging module",
    )
    parser.add_argument(
        "--logging_file",
        type=str,
        help="where to log results, default is stderr",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging_level = getattr(logging, args.logging_level)
    logging.basicConfig(
        level=logging_level,
        filename=args.logging_file,
    )

    encoder_args = []
    if args.model_arg:
        encoder_args.append(args.model_arg)

    encoder = model_mapping(args.model)(*encoder_args)

    model = Predictor(encoder, args.dropout)

    train_dataset = IAPSDataset(dir=args.dataset_dir, crop=args.crop_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_args = argparse.Namespace(
        output_dir=None,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        device=device,
        disable_tqdm=False,
        adam_beta1=0.9,
        adam_epsilon=1e-8,
        adam_beta2=0.99,
        weight_decay=0,
    )

    trainer = EmbeddingsTrainer(
        model,
        train_dataset,
        training_args,
        logging_level=logging_level,
    )
    trainer.train()


if __name__ == "__main__":
    main()
