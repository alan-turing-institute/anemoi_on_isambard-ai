import time

import pytorch_lightning as pl


class StartupTimerCallback(pl.Callback):
    """Logs wall-clock time at key Lightning startup hooks to decompose training startup overhead.

    Only logs from rank 0. T0 is set at callback instantiation (after imports and Hydra config
    loading, but before model init and Lightning setup). To capture total job startup time,
    compare against the first timestamp in the SLURM job log.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.t0 = time.perf_counter()
        self.last = self.t0
        self._first_batch_done = False

    def _log(self, trainer, label: str) -> None:
        if trainer.global_rank != 0:
            return
        now = time.perf_counter()
        print(
            f"[STARTUP] {label:40s}  elapsed={now - self.t0:7.3f}s  delta={now - self.last:7.3f}s",
            flush=True,
        )
        self.last = now

    def setup(self, trainer, pl_module, stage=None) -> None:
        self._log(trainer, "setup (model + data ready)")

    def on_fit_start(self, trainer, pl_module) -> None:
        self._log(trainer, "on_fit_start")

    def on_train_start(self, trainer, pl_module) -> None:
        self._log(trainer, "on_train_start")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self._first_batch_done:
            self._log(trainer, "first batch start")
            self._first_batch_done = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._first_batch_done:
            return
        if batch_idx == 0:
            self._log(trainer, "first batch end (compile done)")
            self._first_batch_done = False  # stop logging after this
