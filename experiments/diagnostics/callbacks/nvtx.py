import torch
import pytorch_lightning as pl


class StepNVTXCallback(pl.Callback):
    def __init__(self, config) -> None:
        super().__init__()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        torch.cuda.nvtx.range_push("step")

    def on_before_backward(self, trainer, pl_module, loss):
        torch.cuda.nvtx.range_push("backward")

    def on_after_backward(self, trainer, pl_module):
        torch.cuda.nvtx.range_pop()  # end backward

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        torch.cuda.nvtx.range_push("optimizer")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        torch.cuda.nvtx.range_pop()  # end optimizer
        torch.cuda.nvtx.range_pop()  # end step
