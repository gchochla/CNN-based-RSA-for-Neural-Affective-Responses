import os
import csv
from typing import List
from PIL import Image
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class IAPSDataset(Dataset):
    def __init__(self, dir, crop=None):
        self.ids, self.labels = self.read_annotations(dir)
        assert len(self.ids) == len(
            set(self.ids)
        ), f"{len(self.ids)}, {len(set(self.ids))}"
        self.images = self.read_images(dir)
        print(len(self.ids), len(self.labels), len(self.images))
        self.image_transformation = self.get_image_transformation(crop)
        self.embeddings = None

    def get_image_transformation(self, crop):
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

    def read_annotations(self, dir):
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

    def read_images(self, dir):
        images = []
        for img_id in self.ids:
            try:
                img_bn = img_id + ".jpg"
                image = Image.open(os.path.join(dir, "images", img_bn))
            except:
                img_bn = img_id + ".JPG"
                image = Image.open(os.path.join(dir, "images", img_bn))

            images.append(image)

        return images

    def __getitem__(self, index):
        if self.embeddings is None:
            image_data = self.image_transformation(self.images[index])
        else:
            image_data = self.embeddings[index]

        return self.ids[index], image_data, self.labels[index]

    def __len__(self):
        return len(self.ids)

    def set_image_embeddings(self, embeddings):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        assert len(embeddings) == len(self.images)
        self.embeddings = embeddings
