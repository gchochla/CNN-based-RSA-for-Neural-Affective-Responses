import os
import argparse
import logging

import torch
from tqdm import tqdm

from project.image_encoders import AVAILABLE_MODELS, model_mapping
from project.datasets import StimuliDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, type=str, help="dataset directory")
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
    parser.add_argument("--crop_size", default=224, type=int, help="dimension to crop images to")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="whether to use pretrained model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to pretrained model (if this wasn't specified during"
        " training, passing `--pretrained` is enough to use pretrained model)",
    )
    parser.add_argument("--embeddings_path", type=str, help="path to save embeddings")
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

    root_dir = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], os.pardir))

    logging_level = getattr(logging, args.logging_level)
    logging.basicConfig(
        level=logging_level,
        filename=args.logging_file,
    )

    encoder_args = []
    if args.model_arg:
        encoder_args.append(args.model_arg)

    encoder = model_mapping(args.model)(*encoder_args)

    if args.model_path is not None or args.pretrained:
        if args.model_path is None:
            args.model_path = os.path.join(
                root_dir,
                "models",
                args.model
                + ("_" + str(args.model_arg) if args.model_arg is not None else "")
                + ".pt",
            )

        logging.info("Loading model from " + args.model_path)
        encoder.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    dataset = StimuliDataset(args.dataset_dir, args.crop_size)

    embeddings = {}
    for _id, image in tqdm(dataset, desc="Embedding..."):
        embeddings[_id] = encoder(image.unsqueeze(0))[0].detach()

    embed_list = [embeddings[str(i + 1)] for i in range(60)]
    embeddings = torch.stack(embed_list, dim=0)
    # pearson corr coefficients
    rdm = torch.corrcoef(embeddings)

    if args.embeddings_path is None:

        args.embeddings_path = os.path.join(
            root_dir,
            "embeddings",
            args.model + ("_" + str(args.model_arg) if args.model_arg is not None else "") + ".pt",
        )

    logging.info("Saving model to " + args.embeddings_path)
    torch.save(rdm, args.embeddings_path)


if __name__ == "__main__":
    main()
