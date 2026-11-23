"""Settings for the active scalar regression models and their training loop."""

from pathlib import PureWindowsPath

from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Read MGS_KAN_* environment variables when constructing a model."""

    model_config = SettingsConfigDict(
        env_prefix="MGS_KAN_", frozen=True, allow_inf_nan=False, dotenv_filtering="only_existing"
    )

    features: PositiveInt = 12
    init_seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    mlp_width: PositiveInt = 24
    single_kernels: PositiveInt = 4
    pair_kernels: PositiveInt = 2
    single_center_range: tuple[float, float] = (-1.5, 1.5)
    single_amplitude_sd: NonNegativeFloat = 0.09
    pair_amplitude_sd: NonNegativeFloat = 0.035
    pair_center_sd: NonNegativeFloat = 0.55
    pair_factor_sd: NonNegativeFloat = 0.08


class TrainingSettings(BaseSettings):
    """Read MGS_KAN_TRAIN_* overrides when configuring a regression run."""

    model_config = SettingsConfigDict(
        env_prefix="MGS_KAN_TRAIN_", frozen=True, allow_inf_nan=False, dotenv_filtering="only_existing"
    )

    seed: int = Field(default=42, ge=0, le=2**64 - 1)
    epochs: PositiveInt = 50
    batch_size: PositiveInt = 64
    learning_rate: PositiveFloat = 0.001
    device: str = Field(default="cpu", pattern=r"^(cpu|auto|mps|cuda(:[1-9][0-9]*|:0)?)$")


DEFAULT_CHECKPOINT_FILE = "best.pt"


class TrackingSettings(BaseSettings):
    """Connection settings for the shared MLflow server."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_", frozen=True, dotenv_filtering="only_existing"
    )

    tracking_uri: str = Field(min_length=1)
    experiment_name: str = Field(default="mgs-kan", min_length=1)
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE

    @field_validator("checkpoint_file")
    @classmethod
    def validate_checkpoint_file(cls, name: str) -> str:
        if not name or name == ".." or "\x00" in name or PureWindowsPath(name).name != name:
            raise ValueError("Expected a checkpoint filename without directories")
        return name
