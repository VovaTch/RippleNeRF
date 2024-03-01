import os
from typing_extensions import Self

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L


class NerfOnlineDataset(Dataset):
    """
    Basic NeRF dataset from two .pkl files from the French guy's mini NeRF implementation.
    """

    def __init__(self, dataset_path: str) -> None:
        """
        Initialize the Dataset class.

        Args:
            dataset_path (str): The path to the dataset.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.data = torch.from_numpy(np.load(dataset_path, allow_pickle=True))

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Retrieve the item at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the following keys:
                - "origin": A torch.Tensor representing the origin of the item.
                - "direction": A torch.Tensor representing the direction of the item.
                - "pixel_value": A torch.Tensor representing the pixel value of the item.
        """
        return {
            "origin": self.data[index, 0:3],
            "direction": self.data[index, 3:6],
            "pixel_value": self.data[index, 6:],
        }


class NerfOnlineDataModule(L.LightningDataModule):
    """
    Lightning data module for the truck NeRF dataset
    """

    def __init__(
        self,
        dataset_folder_path: str,
        batch_size: int = 128,
        val_split: float = 0.01,
        num_workers: int = 0,
    ) -> None:
        """
        Initializes a Dataset object.

        Args:
            dataset_folder_path (str): The path to the dataset folder.
            batch_size (int, optional): The batch size for data loading. Defaults to 128.
            val_split (float, optional): The validation split ratio. Defaults to 0.01.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 0.
        """
        super().__init__()
        self.dataset_folder_path = dataset_folder_path
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """
        Sets up the dataset for training or testing.

        Args:
            stage (str): The stage of the experiment. Can be "fit" for training or None, or "test" for testing.

        Returns:
            None
        """
        if stage == "fit" or stage is None:
            self.complete_dataset = NerfOnlineDataset(
                dataset_path=os.path.join(self.dataset_folder_path, "training_data.pkl")
            )
            self.train_dataset, self.val_dataset = random_split(
                self.complete_dataset,
                [
                    int(len(self.complete_dataset) * (1 - self.val_split)),
                    int(len(self.complete_dataset) * self.val_split),
                ],
            )
        elif stage == "test":
            self.test_dataset = torch.from_numpy(
                np.load(
                    os.path.join(self.dataset_folder_path, "testing_data.pkl"),
                    allow_pickle=True,
                )
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Returns a DataLoader object for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns a validation dataloader.

        Returns:
            EVAL_DATALOADERS: A validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Returns a DataLoader object for the test dataset.

        Returns:
            DataLoader: A DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Initializes a NerfOnlineDataModule object from a configuration object.

        Args:
            cfg (DictConfig): The configuration object.

        Returns:
            Self: The initialized NerfOnlineDataModule object.
        """
        return cls(
            dataset_folder_path=cfg.data.data_path,
            batch_size=cfg.learning.batch_size,
            val_split=cfg.learning.val_split,
            num_workers=cfg.learning.num_workers,
        )
