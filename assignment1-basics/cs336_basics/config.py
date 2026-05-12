from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DataConfig:
    train_data: Path = field(default_factory=lambda: Path("output/tinystories_train.bin"))
    val_data: Path = field(default_factory=lambda: Path("output/tinystories_valid.bin"))
    vocab_path: Path = field(default_factory=lambda: Path("output/tinystories_vocab.json"))
    merge_path: Path = field(default_factory=lambda: Path("output/tinystories_merges.txt"))


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class OptimConfig:
    weight_decay: float = 0.1
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 106667


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "cs336_hw1"
    name: Optional[str] = None


@dataclass
class TrainConfig:
    max_steps: int = 10000
    batch_size: int = 32
    max_l2_norm: float = 1.0
    seed: int = 42
    project: str = "cs336_hw1"
    output_dir: Path = field(default_factory=lambda: Path("output"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("output/checkpoints"))
    log_interval: int = 10
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        def convert_value(key: str, value):
            if key in ("train_data", "val_data", "vocab_path", "merge_path", "output_dir", "checkpoint_dir"):
                return Path(value)
            if key == "betas":
                return tuple(value)
            if key == "rope_theta":
                return float(value)
            if key in ("device", "dtype", "project"):
                return str(value)
            if key == "enabled":
                return bool(value)
            if isinstance(value, str):
                if "." in value or "e" in value.lower():
                    try:
                        return float(value)
                    except ValueError:
                        return value
                try:
                    return int(value)
                except ValueError:
                    return value
            return value

        def convert_section(section_cls, section_dict):
            if section_cls == TrainConfig and "wandb" in section_dict:
                wandb_dict = section_dict.pop("wandb")
                section_dict["wandb"] = convert_section(WandbConfig, wandb_dict)
            kwargs = {k: convert_value(k, v) for k, v in section_dict.items()}
            return section_cls(**kwargs)

        return cls(
            data=convert_section(DataConfig, raw.get("data", {})),
            model=convert_section(ModelConfig, raw.get("model", {})),
            optim=convert_section(OptimConfig, raw.get("optim", {})),
            scheduler=convert_section(SchedulerConfig, raw.get("scheduler", {})),
            train=convert_section(TrainConfig, raw.get("train", {})),
        )

    def validate(self) -> list[str]:
        errors = []

        if self.model.vocab_size <= 0:
            errors.append("model.vocab_size must be positive")
        if self.model.context_length <= 0:
            errors.append("model.context_length must be positive")
        if self.model.d_model <= 0:
            errors.append("model.d_model must be positive")
        if self.model.num_layers <= 0:
            errors.append("model.num_layers must be positive")
        if self.model.num_heads <= 0:
            errors.append("model.num_heads must be positive")
        if self.model.d_ff <= 0:
            errors.append("model.d_ff must be positive")
        if self.model.d_model % self.model.num_heads != 0:
            errors.append("model.d_model must be divisible by model.num_heads")

        if self.optim.weight_decay < 0:
            errors.append("optim.weight_decay must be non-negative")
        if self.optim.eps <= 0:
            errors.append("optim.eps must be positive")
        if not len(self.optim.betas) == 2:
            errors.append("optim.betas must have exactly 2 values")

        if self.scheduler.max_learning_rate <= 0:
            errors.append("scheduler.max_learning_rate must be positive")
        if self.scheduler.min_learning_rate <= 0:
            errors.append("scheduler.min_learning_rate must be positive")
        if self.scheduler.warmup_iters < 0:
            errors.append("scheduler.warmup_iters must be non-negative")
        if self.scheduler.cosine_cycle_iters <= 0:
            errors.append("scheduler.cosine_cycle_iters must be positive")

        if self.train.max_steps <= 0:
            errors.append("train.max_steps must be positive")
        if self.train.batch_size <= 0:
            errors.append("train.batch_size must be positive")
        if self.train.max_l2_norm < 0:
            errors.append("train.max_l2_norm must be non-negative")

        return errors


def get_config(yaml_path: Optional[Path] = None) -> TrainingConfig:
    if yaml_path is None:
        yaml_path = Path(__file__).parent.parent / "config_tinystories.yaml"
    config = TrainingConfig.from_yaml(yaml_path)

    errors = config.validate()
    if errors:
        raise ValueError(f"Config validation failed: {errors}")

    return config
