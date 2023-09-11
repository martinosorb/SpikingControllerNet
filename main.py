from pytorch_lightning.cli import LightningCLI
from ControlledModules import EventControllerNet


if __name__ == "__main__":
    cli = LightningCLI(EventControllerNet)
