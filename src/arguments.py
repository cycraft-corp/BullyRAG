import logging
import pathlib
from typing import Optional, Literal, get_args
from dataclasses import dataclass, field

@dataclass 
class AcceleratorArguments:
    precision: Literal["fp16", "bf16", "fp32"] = field(default="fp16")
    accelerator_type: Literal[
        "CAI_Gemini", "CAI_TorchDDP", "CAI_TorchDDP_FP16", "CAI_ZeRO2"
    ] = field(
        default="CAI_Gemini",
        metadata={"aliases": "--accelerator-type"}
    )

@dataclass
class CriterionArguments:
    criterion_name: str = field(
        metadata={
            "aliases": "--criterion-name"
        },
        default="InfoCriterion"
    )
    temperature: float = field(
        metadata={
            "aliases": "--temperature"
        },
        default=0.05
    )

@dataclass
class DataArguments:
    path_to_data: str = field(
        metadata={
            "aliases": "--path-to-data",
            "required": True,
        }
    )
    path_to_val_data: str = field(
        metadata={
            "aliases": "--path-to-val-data"
        },
        default=""
    )
    dataset_name: str = field(
        metadata={
            "aliases": "--dataset-name"
        },
        default="BaseDataset"
    )

"""file copied from USE repo"""
@dataclass
class ProviderArguments:
    default_prefix: str = field(
        default="",
        metadata={
            "aliases": "--default-prefix"
        }
    )
    default_suffix: str = field(
        default="",
        metadata={
            "aliases": "--default-suffix"
        }
    )

    incontext_prefix_mode: str = field(
        default="empty",
        metadata={"aliases": "--incontext-prefix-mode"}
    )
    incontext_suffix_mode: str = field(
        default="empty",
        metadata={"aliases": "--incontext-suffix-mode"}
    )

@dataclass
class TrainingArguments:
    title: str = field(metadata={'required': True})
    path_to_checkpoint_dir: pathlib.Path = field(
        metadata={
            "aliases": "--path-to-checkpoint-dir",
            "required": True
        }
    )
    lr: float = field(default=5e-6)
    epochs: int = field(default=1)
    shuffle: bool = field(default=False)

    lr_scheduler_type: str = field(
        default="linear",
        metadata={"aliases": "--lr-scheduler-type"}
    )
    warmup_ratio: float = field(
        default=0,
        metadata={"aliases": "--warmup-ratio"}
    )

    per_device_train_batch_size: int = field(
        default=32,
        metadata={
            'aliases': ['--batch-size', '--batch_size', '--per-device-train-batch-size'],
            'help': 'The batch size per GPU/TPU core/CPU for training.'
        }
    )

    log_level: str = field(
        default="INFO",
        metadata={
            "aliases": "--log-level",
            "help": f"Set logging level. Choices=[{'|'.join(logging._nameToLevel.keys())}]"
        }
    )
    log_interval: int = field(
        default=1,
        metadata={'aliases': '--log-interval'}
    )

    checkpoint_interval: int = field(
        default=10000,
        metadata={"aliases": "--checkpoint-interval"}
    )
    device: str = field(default="cuda")
    
    def __post_init__(self):
        self.log_level = logging._nameToLevel[self.log_level.upper()]
        self.path_to_checkpoint_dir /= self.title

@dataclass
class EvaluatorArguments:
    pipeline_name: str = field(
        default="EmbeddingPipeline",
        metadata={
            "aliases": "--pipeline-name"
        }
    )

@dataclass
class ModelArguments:
    model_name: str = field(
        default="BaseModel",
        metadata={
            "aliases": "--model-name",
        }
    )

    model_max_length: int = field(
        default=512,
        metadata={
            "aliases": ["--max-sequence-len", "--max_sequence_len", "--model-max-length"],
            "help": f"The maximum sequence length of the tokenizer"
        }
    )

    path_to_model_weight: str = field(
        default=None,
        metadata={
            "aliases": ["--path-to-model-weight", "--huggingface-repo-name"],
            "help": f"The path to the huggingface-style model weight directory or the huggingface repo name"
        }
    )

    def __post_init__(self):
        if self.path_to_model_weight is None:
            if self.model_name == "Bert":
                self.path_to_model_weight = "bert-base-uncased"
