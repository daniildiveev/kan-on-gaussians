"""Model settings shared by the active scalar baselines."""

from pydantic import NonNegativeFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """Read MGS_KAN_* environment variables when constructing a model."""

    model_config = SettingsConfigDict(env_prefix="MGS_KAN_", frozen=True, allow_inf_nan=False)

    features: PositiveInt = 12
    mlp_width: PositiveInt = 24
    single_kernels: PositiveInt = 4
    pair_kernels: PositiveInt = 2
    single_center_range: tuple[float, float] = (-1.5, 1.5)
    single_amplitude_sd: NonNegativeFloat = 0.09
    pair_amplitude_sd: NonNegativeFloat = 0.035
    pair_center_sd: NonNegativeFloat = 0.55
    pair_factor_sd: NonNegativeFloat = 0.08
