"""Training and experiment orchestration."""

from core.callbacks import BestMetricTracker, CheckpointCallback
from core.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointContract,
    CheckpointContractError,
    CheckpointLoadResult,
    current_git_commit,
    checkpoint_compatibility_config,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_contract,
)
from core.lightning_module import SketchformerTrainingModule
from core.losses import SketchformerLoss, SketchformerLossOutput, masked_mean
from core.metrics import ReconstructionMetrics, reconstruction_metrics
from core.seed import seed_worker, set_seed
from core.trainer import StepResult, average_logs, move_to_device, train_step, validation_step

__all__ = [
    "BestMetricTracker",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointContract",
    "CheckpointContractError",
    "CheckpointLoadResult",
    "CheckpointCallback",
    "ReconstructionMetrics",
    "SketchformerLoss",
    "SketchformerLossOutput",
    "SketchformerTrainingModule",
    "StepResult",
    "average_logs",
    "current_git_commit",
    "checkpoint_compatibility_config",
    "latest_checkpoint",
    "load_checkpoint",
    "masked_mean",
    "move_to_device",
    "reconstruction_metrics",
    "save_checkpoint",
    "seed_worker",
    "set_seed",
    "train_step",
    "validation_step",
    "validate_checkpoint_contract",
]
