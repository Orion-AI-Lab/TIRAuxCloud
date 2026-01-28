from typing import Tuple

import torch

import src.models.base_module


class SAMLitModule(src.models.base_module.BaseLitModule):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

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
        logits ,iou_preds = self.forward(x)
        loss = self.hparams.criterion(logits ,y, iou_preds)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
