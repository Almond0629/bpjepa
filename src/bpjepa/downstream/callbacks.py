"""Checkpoint cadence for fine-tuning and frozen-head training."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import Callback, LightningModule, Trainer


class EpochCheckpoint(Callback):
    """Save every N epochs, after validation on validation epochs."""

    def __init__(
        self,
        dirpath: str | Path,
        every_n_epochs: int = 5,
        validation_every_n_epochs: int = 10,
        save_last: bool = True,
    ) -> None:
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if validation_every_n_epochs < 1:
            raise ValueError("validation_every_n_epochs must be positive")
        if validation_every_n_epochs % every_n_epochs:
            raise ValueError("validation cadence must be a multiple of checkpoint cadence")
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs
        self.validation_every_n_epochs = validation_every_n_epochs
        self.save_last = save_last

    def _completed_epoch(self, trainer: Trainer) -> int:
        return trainer.current_epoch + 1

    def _save(self, trainer: Trainer) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        epoch_path = self.dirpath / f"epoch={trainer.current_epoch}.ckpt"
        trainer.save_checkpoint(str(epoch_path), weights_only=False)
        if self.save_last:
            trainer.save_checkpoint(str(self.dirpath / "last.ckpt"), weights_only=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        completed = self._completed_epoch(trainer)
        if completed % self.every_n_epochs == 0 and completed % self.validation_every_n_epochs:
            self._save(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        completed = self._completed_epoch(trainer)
        if completed % self.validation_every_n_epochs == 0:
            self._save(trainer)
