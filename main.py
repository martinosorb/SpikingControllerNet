from pytorch_lightning.cli import LightningCLI
from ControlledModules import DiffControllerNet
from data import MNISTDataModule


if __name__ == "__main__":
    cli = LightningCLI(DiffControllerNet, datamodule_class=MNISTDataModule)
