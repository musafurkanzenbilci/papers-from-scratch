from pydantic import BaseModel
from typing import Optional

class DataConfig(BaseModel):
    name: str
    val_ratio: float
    batch_size: int

class ModelConfig(BaseModel):
    name: str
    init: Optional[str] = None
    init_std: Optional[float] = None
    init_bias: Optional[float] = None

class OptimConfig(BaseModel):
    name: str
    lr: float
    momentum: float
    weight_decay: float

class TrainConfig(BaseModel):
    epochs: int
    early_stop_patience: int
    # metric: int
    out_dir: str


class RunConfig(BaseModel):
    seed: int
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig