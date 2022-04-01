import argparse
import os

import torch
from tqdm import tqdm

from project.datasets import IAPSDataset, StimuliDataset
from project.image_encoders import ResNetEmbeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iaps_dataset_dir",
        required=True,
        type=str,
        help="directory of IAPS images",
    )
    parser.add_argument(
        "--stimuli_dir",
        required=True,
        type=str,
        help="directory of stimuli images",
    )
    parser.add_argument(
        "--retrieved",
        default=10,
        type=int,
        help="how many images from IAPS to retrieve for each stimulus",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    iaps = IAPSDataset(args.iaps_dataset_dir)
    stimuli = StimuliDataset(args.stimuli_dir)

    encoder = ResNetEmbeddings()
    encoder.eval()

    iaps_embeddings = {}
    stimuli_embeddings = {}

    with torch.no_grad():
        for _id, img, _ in tqdm(iaps, desc="Embedding IAPS..."):
            iaps_embeddings[_id] = encoder(img.unsqueeze(0))
        for _id, img in tqdm(stimuli, desc="Embedding stimuli..."):
            stimuli_embeddings[_id] = encoder(img.unsqueeze(0))

    retrievals = {}

    for stimulus in tqdm(stimuli_embeddings, desc="Retrieving..."):
        similarities = {
            _id: torch.nn.functional.cosine_similarity(
                stimuli_embeddings[stimulus], iaps_embeddings[_id]
            )
            for _id in iaps_embeddings
        }

        similarities = {
            k: v
            for i, (k, v) in enumerate(
                sorted(
                    similarities.items(), key=lambda item: item[1], reverse=True
                )
            )
            if i < args.retrieved
        }

        retrievals[stimulus] = similarities

    retrieval_dir = os.path.join(
        args.stimuli_dir, os.path.pardir, "stimuli_retrieval"
    )
    if not os.path.exists(retrieval_dir):
        os.makedirs(retrieval_dir)

    for stimulus in tqdm(retrievals, desc="Saving..."):
        stimulus_dir = os.path.join(retrieval_dir, stimulus)
        if not os.path.exists(stimulus_dir):
            os.makedirs(stimulus_dir)

        image_idx = stimuli.ids.index(stimulus)
        stimulus_image = stimuli.images[image_idx]
        stimulus_image.save(os.path.join(stimulus_dir, "0000.jpg"))

        for iaps_id in retrievals[stimulus]:
            image_idx = iaps.ids.index(iaps_id)
            iaps_image = iaps.images[image_idx]
            iaps_image.save(os.path.join(stimulus_dir, iaps_id + ".jpg"))


if __name__ == "__main__":
    main()
