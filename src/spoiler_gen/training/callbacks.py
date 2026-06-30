import glob
import os
import shutil

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from spoiler_gen.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DiskSpaceGuardCallback(TrainerCallback):
    """Callback to guard disk space by deleting older checkpoints.

    Keeps only the latest checkpoint to respect Kaggle's 20GB output cap.
    """

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        """Deletes older checkpoints in output_dir, leaving only the most recent one."""
        output_dir = args.output_dir
        if not output_dir or not os.path.exists(output_dir):
            return control

        # Log current disk usage for diagnostics
        try:
            total, used, free = shutil.disk_usage(output_dir)
            logger.info(
                f"Disk Usage -> Total: {total / (1024**3):.2f}GB, "
                f"Used: {used / (1024**3):.2f}GB, "
                f"Free: {free / (1024**3):.2f}GB"
            )
        except Exception as e:
            logger.warning(f"Failed to fetch disk usage metrics: {e}")

        # Find all checkpoint- directories
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if len(checkpoints) <= 1:
            return control

        # Sort checkpoints by step number (checkpoint-100, checkpoint-200, etc.)
        def extract_step(path: str) -> int:
            try:
                return int(os.path.basename(path).split("-")[-1])
            except ValueError:
                return -1

        checkpoints.sort(key=extract_step)

        # Keep only the latest checkpoint, delete the rest
        keep_checkpoint = checkpoints[-1]
        for ckpt in checkpoints[:-1]:
            try:
                logger.info(f"DiskSpaceGuard deleting old checkpoint: {ckpt}")
                shutil.rmtree(ckpt)
            except Exception as e:
                logger.error(f"Failed to delete old checkpoint directory {ckpt}: {e}")

        return control
