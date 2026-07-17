from trainer.base import BaseTrainer, TrainingState
from trainer.supervised import SupervisedTrainer

TRAINERS = {
    "supervised": SupervisedTrainer,
}


def build_trainer(trainer_type: str, **kwargs) -> BaseTrainer:
    """Construct a trainer by registered name, forwarding all keyword arguments."""
    if trainer_type not in TRAINERS:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. Available: {', '.join(TRAINERS)}"
        )
    return TRAINERS[trainer_type](**kwargs)
