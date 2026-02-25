import lightning.pytorch as pl
from lightning.pytorch import Callback


import os.path
import numpy
import torch
from typing import Sequence, Any, Dict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from torchvision.utils import make_grid

from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info



class SaveImagesHook(Callback):
    def __init__(
        self,
        save_dir="val",
        save_compressed=False,
        log_preview=True,
        preview_max_images=16,
        preview_ncols=4,
    ):
        self.save_dir = save_dir
        self.save_compressed = save_compressed
        self.log_preview = log_preview
        self.preview_max_images = preview_max_images
        self.preview_ncols = preview_ncols

    def save_start(self, target_dir):
        self.samples = []
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            if os.listdir(target_dir) and "debug" not in str(target_dir):
                raise FileExistsError(f'{self.target_dir} already exists and not empty!')
        rank_zero_info(f"Save images to {self.target_dir}")
        self._saved_num = 0

    def save_image(self, trainer, pl_module, images, metadatas,):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for sample, metadata in zip(images, metadatas):
            save_fn = metadata.pop("save_fn", None)
            self.executor_pool.submit(save_fn, sample, metadata, self.target_dir)

    def process_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        xT, y, metadata = batch
        b, c, h, w = samples.shape
        if not self.save_compressed or self._saved_num < 10:
            self._saved_num += b
            self.save_image(trainer, pl_module, samples, metadata)

        all_samples = pl_module.all_gather(samples).view(-1, c, h, w)
        if trainer.is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples.append(all_samples)

    def _build_preview_grid(self):
        if len(self.samples) == 0:
            return None, None
        all_samples = numpy.concatenate(self.samples, axis=0)
        preview = all_samples[: self.preview_max_images]
        preview_tensor = torch.from_numpy(preview).permute(0, 3, 1, 2).float() / 255.0
        grid = make_grid(preview_tensor, nrow=self.preview_ncols)
        grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return grid, grid_uint8

    def _log_preview(self, trainer, stage, grid, grid_uint8, image_path):
        if not self.log_preview:
            return
        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            logger = getattr(trainer, "logger", None)
            loggers = [logger] if logger is not None else []
        tag = f"{stage}/preview"
        for logger in loggers:
            if logger is None:
                continue
            logger_name = logger.__class__.__name__.lower()
            if "wandb" in logger_name:
                try:
                    import wandb

                    logger.experiment.log(
                        {
                            tag: wandb.Image(image_path),
                            "global_step": trainer.global_step,
                        }
                    )
                except Exception as e:
                    rank_zero_info(f"[save_images] skip wandb preview logging: {e}")
            elif "tensorboard" in logger_name:
                try:
                    logger.experiment.add_image(tag, grid, global_step=trainer.global_step, dataformats="CHW")
                except Exception as e:
                    rank_zero_info(f"[save_images] skip tensorboard preview logging: {e}")

    def save_end(self, trainer=None, stage="val"):
        preview_path = None
        if self.save_compressed and len(self.samples) > 0:
            samples = numpy.concatenate(self.samples)
            numpy.savez(f'{self.target_dir}/output.npz', arr_0=samples)
        if len(self.samples) > 0:
            grid, grid_uint8 = self._build_preview_grid()
            if grid is not None:
                preview_path = os.path.join(self.target_dir, f"{stage}_preview_step{trainer.global_step if trainer else 0}.png")
                Image.fromarray(grid_uint8).save(preview_path)
                rank_zero_info(f"Save preview grid to {preview_path}")
                if trainer is not None:
                    self._log_preview(trainer, stage, grid, grid_uint8, preview_path)
        self.executor_pool.shutdown(wait=True)
        self.target_dir = None
        self.executor_pool = None
        self.samples = []

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end(trainer=trainer, stage="val")

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, "predict")
        self.save_start(target_dir)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, samples, batch)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end(trainer=trainer, stage="predict")

    def state_dict(self) -> Dict[str, Any]:
        return dict()
