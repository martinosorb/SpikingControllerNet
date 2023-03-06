from pytorch_lightning.cli import LightningCLI
from ControlledModules import EventControllerNet
from data import MNISTDataModule


if __name__ == "__main__":
    cli = LightningCLI(EventControllerNet, datamodule_class=MNISTDataModule)
