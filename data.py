import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms as tr
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize((0.1307,), (0.3081,)), 
                torch.flatten
            ],
        )

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform, download=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )


class N_MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data/", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = 8

    # @staticmethod
    # def transform_frames(x):
    #     # frames are TPHW
    #     print(x.shape)
    #     x = torch.tensor(x)
    #     x = torch.moveaxis(x, 2, 0)
    #     print(x.shape)
    #     x = torch.flatten(x, start_dim=1)
    #     print(x.shape)
    #     raise
    #     return x

    def setup(self, stage: str):
        from tonic.datasets import NMNIST
        from tonic.transforms import ToFrame

        # This leads to 32 time bins. For NMNIST, this will mean ~10 ms per bin.
        # Sensor size is 34 and 2 polarities: size is 34*34*2=2312
        to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=32)
        transform = tr.Compose([to_frame, torch.tensor, torch.nn.Flatten(start_dim=1)])

        self.nmnist_test = NMNIST(self.data_dir, train=False, transform=transform)
        nmnist_full = NMNIST(self.data_dir, train=True, transform=transform)
        self.nmnist_train, self.nmnist_val = random_split(nmnist_full, [.9, .1])

    def train_dataloader(self):
        return DataLoader(
            self.nmnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.nmnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.nmnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
