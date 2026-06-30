import os

from pydantic import BaseModel, field_validator

from spoiler_gen.utils.io_utils import read_yaml


class BaseConfig(BaseModel):
    project_name: str
    random_seed: int
    log_level: str
    output_root: str


class DataConfig(BaseModel):
    raw_dir: str
    interim_dir: str
    processed_dir: str
    train_file: str
    val_file: str
    test_file: str | None = None
    max_input_length: int
    max_target_length: int
    input_template: str
    spoiler_join_token: str


class TrainConfig(BaseModel):
    model_name_or_path: str
    seeds: list[int]
    learning_rate: float
    lr_scheduler_type: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    fp16: bool
    generation_max_length: int
    generation_num_beams: int

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_steps must be a positive integer greater than 0")
        return v

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("seeds list cannot be empty")
        return v


class AppConfig(BaseModel):
    base: BaseConfig
    data: DataConfig
    train: TrainConfig


def load_config(config_dir: str) -> AppConfig:
    """Load and validate configuration from YAML files in a directory.

    Args:
        config_dir: Directory containing base.yaml, data.yaml, and train.yaml.

    Returns:
        An AppConfig instance containing validated configurations.
    """
    base_path = os.path.join(config_dir, "base.yaml")
    data_path = os.path.join(config_dir, "data.yaml")
    train_path = os.path.join(config_dir, "train.yaml")

    base_dict = read_yaml(base_path)
    data_dict = read_yaml(data_path)
    train_dict = read_yaml(train_path)

    return AppConfig(
        base=BaseConfig(**base_dict), data=DataConfig(**data_dict), train=TrainConfig(**train_dict)
    )
