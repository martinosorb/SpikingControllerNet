from pytorch_lightning.cli import LightningCLI
from ControlledLayer import ControlledNetwork
from data import MNISTDataModule


if __name__ == "__main__":
    cli = LightningCLI(ControlledNetwork, datamodule_class=MNISTDataModule)
