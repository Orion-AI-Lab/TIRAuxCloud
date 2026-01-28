from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule

from mmeval import MeanIoU

class BaseLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.train_miou = MeanIoU(num_classes=self.hparams.num_classes,classwise_results=True)
        self.val_miou = MeanIoU(num_classes=self.hparams.num_classes,classwise_results=True)
        self.test_miou = MeanIoU(num_classes=self.hparams.num_classes,classwise_results=True)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_miou.reset()
        self.val_miou.reset()
        self.test_miou.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch["img"], batch["ann"]
        logits = self.forward(x)
        loss = self.hparams.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)
        
        # print(preds.shape) # (8, 256, 256)
        # print(targets.shape) # (8, 256, 256)

        # update and log metrics
        batch_res = self.train_miou(preds, targets)

        
        
        self.log("train/accuracy", batch_res['mAcc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", batch_res['mPrecision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", batch_res['mRecall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1score", batch_res['mFscore'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/miou", batch_res['mIoU'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", batch_res['mDice'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kappa", batch_res['kappa'], on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        batch_res = self.val_miou(preds, targets)
        self.val_miou.add(preds, targets)
        self.log("val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", batch_res['mAcc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", batch_res['mPrecision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", batch_res['mRecall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1score", batch_res['mFscore'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/miou", batch_res['mIoU'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", batch_res['mDice'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kappa", batch_res['kappa'], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        batch_res = self.val_miou.compute()
        self.val_miou.reset()
        self.log("val/accuracy", batch_res['mAcc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", batch_res['mPrecision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", batch_res['mRecall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1score", batch_res['mFscore'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/miou", batch_res['mIoU'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", batch_res['mDice'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kappa", batch_res['kappa'], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        batch_res = self.test_miou(preds, targets)
        self.test_miou.add(preds, targets)
        
        
        self.log("test/accuracy", batch_res['mAcc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", batch_res['mPrecision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", batch_res['mRecall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1score", batch_res['mFscore'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/miou", batch_res['mIoU'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", batch_res['mDice'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/kappa", batch_res['kappa'], on_step=False, on_epoch=True, prog_bar=True)
        
        
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        batch_res = self.test_miou.compute()
        self.test_miou.reset()
        self.log("test/accuracy", batch_res['mAcc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", batch_res['mPrecision'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", batch_res['mRecall'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1score", batch_res['mFscore'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/miou", batch_res['mIoU'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice", batch_res['mDice'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/kappa", batch_res['kappa'], on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BaseLitModule(None, None, None, None)
