import os
import csv
from typing import Optional, Union, Dict, Tuple, List
from PIL import Image, ImageFile

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


class IAPSDataset(Dataset):
    """Dataset for IAPS images and VAD annotations.

    Attributes:
        ids: IDs of individual examples (even though
            most are integers, some have different
            subversions whose IDs resemble floats)
        labels: VAD labels in the [0, 1] range.
        images: list of PIL images.
        image_transformation: transformation from PIL
            to Tensor (plus preprocessing).
        embeddings: image embeddings (to be provided
            after the initialization of the class)
    """

    def __init__(
        self,
        dir: str,
        exclude_fn: Optional[str] = None,
        crop: Optional[int] = None,
    ):
        """Init.

        Args:
            dir: root directory of dataset (should contain
                directory of images and label file).
            exclude_fn: file that contains example IDs to
                be excluded. Each ID should be in a separate
                line and at the end of it w.r.t whitespaces.
            crop: size to crop the images when fetching them
                through.
        """
        self.ids, self.labels = self.read_annotations(dir)
        self.images = self.read_images(dir)
        self.image_transformation = self.get_image_transformation(crop)
        self.embeddings = None

        if exclude_fn is not None:
            self.filter(exclude_fn)

    def get_image_transformation(
        self, crop: Union[int, None]
    ) -> "torchvision.Transform":
        """Creates and return image transformation.

        Args:
            crop: size to crop the images when fetching them
                through.
        """
        if crop is None:
            crop = 224

        image_transform = transforms.Compose(
            [
                transforms.Resize(crop + 32),
                transforms.RandomCrop(crop),
                # transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        return image_transform

    def read_annotations(self, dir: str) -> Tuple[List[str], np.ndarray]:
        """Reads VAD annotations.

        Args:
            dir: parent directory of dataset.

        Returns:
            IDs and corresponding labels.
        """

        def normalize(rating):
            return (rating - 1) / (9 - 1)

        labels = []
        ids = []

        with open(os.path.join(dir, "AllSubjects_1-20.txt")) as fp:
            reader = csv.reader(fp, delimiter="\t")
            _ = next(reader)  # headers

            for row in reader:
                id = row[1]
                # dataset has some images labeled twice, with a different "set"
                # handle them like this for now
                if id not in ids:
                    valence = normalize(float(row[2]))
                    arousal = normalize(float(row[4]))
                    dominance = normalize(
                        float(row[6]) if row[6] != "." else float(row[8])
                    )

                    ids.append(id)
                    labels.append((valence, arousal, dominance))

        return ids, np.array(labels, dtype=np.float32)

    def read_images(self, dir: str) -> List[Image.Image]:
        """Reads images.

        Args:
            dir: parent directory of dataset.

        Returns:
            A list of images (same order as `ids` attribute).
        """
        images = []
        to_tensor = transforms.ToTensor()
        for img_id in self.ids:
            try:
                img_bn = img_id + ".jpg"
                image = Image.open(os.path.join(dir, "images", img_bn))
            except:
                img_bn = img_id + ".JPG"
                image = Image.open(os.path.join(dir, "images", img_bn))

            images.append(to_tensor(image))

        return images

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, np.ndarray]:
        """Returns id, image (or embeddings, if provided)
        and the labels."""
        if self.embeddings is None:
            image_data = self.image_transformation(self.images[index])
        else:
            image_data = self.embeddings[index]

        return self.ids[index], image_data, self.labels[index]

    def __len__(self) -> int:
        return len(self.ids)

    def set_image_embeddings(self, embedding_dict: Dict[str, np.ndarray]):
        """Sets embeddings of ALL images to be used instead
        of the images themselves durign fetching.

        Args:
            embedding_dict: embeddings index by ID.
        """

        assert len(embeddings) == len(self.images)

        embeddings = [None for _ in range(len(self))]
        for _id, embedding in embedding_dict.items():
            idx = self.ids.index(_id)
            embeddings[idx] = embedding
        self.embeddings = embeddings

    def filter(self, exclude_filename: str):
        """Filters out examples that are included in
        the exclude file provided."""
        with open(exclude_filename) as fp:
            exclude_ids = [line.split()[-1].strip() for line in fp.readlines()]

        # descending order to avoid having to correct indices
        exclude_inds = sorted(
            [self.ids.index(_id) for _id in exclude_ids], reverse=True
        )
        # there's probably a better way to do this
        include_inds = [i for i in range(len(self)) if i not in exclude_ids]

        for idx in exclude_inds:
            self.ids.pop(idx)
            self.images.pop(idx)

        self.labels = self.labels[include_inds]


class StimuliDataset(Dataset):
    """Basically an image dataset.

    Attributes:
        ids: IDs of individual examples.
        images: list of images.
        image_transformation: transformation from PIL
            to Tensor (plus preprocessing).
    """

    def __init__(self, dir: str, crop: Optional[int] = None):
        """Init.

        Args:
            dir: directory of images.
            crop: size to crop the images when fetching them
                through.
        """
        self.ids, self.images = self.read_images(dir)
        self.image_transformation = self.get_image_transformation(crop)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        """Returns ID and preprocessed image."""
        return self.ids[index], self.image_transformation(self.images[index])

    def __len__(self) -> int:
        return len(self.images)

    def get_image_transformation(
        self, crop: Union[int, None]
    ) -> "torchvision.Transform":
        """Creates and return image transformation.

        Args:
            crop: size to crop the images when fetching them
                through.

        Returns:
            Image transformation.
        """

        if crop is None:
            crop = 224

        image_transform = transforms.Compose(
            [
                transforms.Resize(crop + 32),
                transforms.RandomCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        return image_transform

    def read_images(self, dir: str) -> Tuple[List[str], List[Image.Image]]:
        """Reads images from directory.

        Args:
            dir: image directory

        Returns:
            IDs and the corresponding images.
        """
        images = []
        ids = []
        for img_bn in os.listdir(dir):
            ids.append(os.path.splitext(img_bn)[0])
            images.append(Image.open(os.path.join(dir, img_bn)))
        return ids, images
