from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    name: str
    val_ratio: float
    batch_size: int

class ModelConfig(BaseModel):
    name: str
    init: str
    init_std: float
    init_bias: float

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