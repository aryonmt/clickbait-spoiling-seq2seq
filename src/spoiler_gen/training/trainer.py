import os
import shutil

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from spoiler_gen.config import AppConfig
from spoiler_gen.modeling.seq2seq_model import load_model_and_tokenizer
from spoiler_gen.training.callbacks import DiskSpaceGuardCallback
from spoiler_gen.utils.logging_utils import get_logger
from spoiler_gen.utils.seed_utils import set_all_seeds

logger = get_logger(__name__)


def build_training_args(app_cfg: AppConfig, output_dir: str, seed: int) -> Seq2SeqTrainingArguments:
    """Map training configuration to HuggingFace Seq2SeqTrainingArguments.

    Args:
        app_cfg: Top-level AppConfig object.
        output_dir: Output directory path for saving checkpoints.
        seed: Seed value for model initialization and training.

    Returns:
        Seq2SeqTrainingArguments instance.
    """
    train_cfg = app_cfg.train

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        max_steps=train_cfg.max_steps,
        eval_steps=train_cfg.eval_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        fp16=train_cfg.fp16,
        predict_with_generate=True,
        generation_max_length=train_cfg.generation_max_length,
        generation_num_beams=train_cfg.generation_num_beams,
        seed=seed,
        data_seed=seed,
        evaluation_strategy="steps" if train_cfg.eval_steps > 0 else "no",
        save_strategy="steps" if train_cfg.save_steps > 0 else "no",
        logging_steps=50,
        report_to=["none"],  # Avoid wandb prompting/hanging in Kaggle non-interactive environments
        weight_decay=0.01,
        warmup_ratio=0.1,
    )


def build_trainer(
    model, tokenizer, train_ds, eval_ds, training_args, compute_metrics_fn=None
) -> Seq2SeqTrainer:
    """Construct Seq2SeqTrainer attached with custom DiskSpaceGuardCallback.

    Args:
        model: Model object.
        tokenizer: Tokenizer object.
        train_ds: Training dataset.
        eval_ds: Evaluation dataset.
        training_args: Seq2SeqTrainingArguments.
        compute_metrics_fn: Optional metrics computation function.

    Returns:
        Seq2SeqTrainer instance.
    """
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
        callbacks=[DiskSpaceGuardCallback()],
    )


def train_one_seed(
    seed: int, app_cfg: AppConfig, train_ds, eval_ds, compute_metrics_fn=None
) -> str:
    """Orchestrate training of a model for a single seed.

    Loads the base model, configures paths, runs training, and saves
    exclusively the final model checkpoint to save disk space.

    Args:
        seed: Random seed for this specific model run.
        app_cfg: AppConfig object containing all paths and hyperparams.
        train_ds: Training dataset.
        eval_ds: Evaluation dataset.
        compute_metrics_fn: Optional metrics callback.

    Returns:
        Path to the saved final model checkpoint.
    """
    # 1. Guarantee strict reproducibility before model instantiation
    set_all_seeds(seed)
    logger.info(f"Set all random seeds to: {seed}")

    # 2. Configure output paths
    seed_output_dir = os.path.join(
        app_cfg.base.output_root, "checkpoints", f"flan-t5-large-seed{seed}"
    )
    os.makedirs(seed_output_dir, exist_ok=True)

    # 3. Load model and tokenizer
    logger.info(f"Loading base seq2seq model: {app_cfg.train.model_name_or_path}")
    model, tokenizer = load_model_and_tokenizer(app_cfg.train.model_name_or_path)

    # 4. Compile training configurations
    training_args = build_training_args(app_cfg, seed_output_dir, seed)

    # 5. Build Trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        training_args=training_args,
        compute_metrics_fn=compute_metrics_fn,
    )

    # 6. Execute Training
    logger.info(f"Starting model training for seed {seed}...")
    trainer.train()

    # 7. Save only the final model checkpoint
    final_save_path = os.path.join(seed_output_dir, "final_model")
    logger.info(f"Saving final trained model to: {final_save_path}")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)

    # Defensive clean up: remove any intermediate checkpoint-X folders left in output_dir
    for item in os.listdir(seed_output_dir):
        item_path = os.path.join(seed_output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            logger.info(f"Cleaning up post-training leftover checkpoint: {item_path}")
            try:
                shutil.rmtree(item_path)
                logger.info(f"Successfully deleted leftover checkpoint: {item_path}")
            except Exception as e:
                logger.error(f"Failed to delete leftover checkpoint {item_path}: {e}")

    return final_save_path
