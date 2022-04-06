import logging
from argparse import Namespace
from typing import Dict, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from project.predictors import Predictor
from project.datasets import IAPSDataset


def get_linear_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: Optional[int] = -1,
) -> optim.lr_scheduler.LambdaLR:
    """
    From: `https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/optimization.py#L75`.
    Create a schedule with a learning rate that decreases linearly from
    the initial lr set in the optimizer to 0, after a warmup period during
    which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer:
            The optimizer for which to schedule the learning rate.
        num_warmup_steps:
            The number of steps for the warmup phase.
        num_training_steps:
            The total number of training steps.
        last_epoch:
            The index of the last epoch when resuming training.

    Return:
        Appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class EmbeddingsTrainer:
    """Image encoder (aka image embeddings) trainer class.

    Attributes:
        model: predictor model.
        train_dataset: train dataset.
        args: various training and evaluation arguments.
        dev_dataset: evaluation dataset.
        logger: logger.
    """

    def __init__(
        self,
        model: Predictor,
        dataset: IAPSDataset,
        train_args: Namespace,
        dev_dataset: Optional[IAPSDataset] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            model: entire predictor.
            dataset: train dataset.
            train_args: whichever namespace with training arguments
                (device, learning_rate, adam_beta1, adam_beta2,
                adam_epsilon, weight_decay, train_batch_size,
                eval_batch_size, etc.)
            dev_dataset: evaluation dataset.
            logging_level: at which level to log messages.
        """
        self.model = model
        self.train_dataset = dataset
        self.args = train_args
        self.dev_dataset = dev_dataset
        self.do_eval = dev_dataset is not None

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    def train(self):
        """Trains `model` on `train_dataset`."""
        self.model = self.model.to(self.args.device)
        self.model.train()

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=[self.args.adam_beta1, self.args.adam_beta2],
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
        )
        if self.do_eval:
            dev_data_loader = DataLoader(
                self.dev_dataset, batch_size=self.args.eval_batch_size
            )

        num_steps = int(len(data_loader) * self.args.num_train_epochs)
        warmup_steps = int(self.args.warmup_ratio * num_steps)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_steps,
        )

        for epoch in range(int(self.args.num_train_epochs)):
            train_loss = 0.0
            for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}: Training"):
                ids, data, labels = batch
                data = data.to(self.args.device)
                labels = labels.to(self.args.device)

                predictions = self.model(data)

                loss = self.criterion(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                train_loss += loss.item() * len(data)

            train_loss /= len(self.train_dataset)

            results = dict(train_loss=train_loss)

            if self.do_eval:
                results.update(
                    self.evaluate(
                        dev_data_loader, f"Epoch {epoch+1}: Evaluation"
                    )
                )

            self.logger.info(f"Epoch {epoch+1} metrics: {results}")

    def evaluate(self, data_loader, desc) -> Dict[str, float]:
        """Evaluates on provided `data_loader`.

        Args:
            data_loader: `DataLoader` to evaluate on.
            desc: description used in `tqdm`.

        Returns:
            A dict of eval metrics.
        """
        self.model.eval()

        eval_loss = 0.0
        for batch in tqdm(data_loader, desc=desc):
            ids, data, labels = batch
            data = data.to(self.args.device)
            labels = labels.to(self.args.device)
            predictions = self.model(data)

            loss = self.criterion(predictions, labels)
            eval_loss += loss.item() * len(data)
        eval_loss /= len(data_loader.dataset)

        self.model.train()

        return dict(eval_loss=eval_loss)
