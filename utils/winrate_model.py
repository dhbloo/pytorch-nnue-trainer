import numpy as np


class WinrateModel:
    def __init__(self, scaling_factor=200, eval_mate_threshold=29000, soft_mate_winrate=False):
        self.scaling_factor = scaling_factor
        self.eval_mate_threshold = eval_mate_threshold
        self.soft_mate_winrate = soft_mate_winrate

    def eval_to_winrate(self, eval: int) -> float:
        if not self.soft_mate_winrate:
            if eval >= self.eval_mate_threshold:
                return 1.0
            elif eval <= -self.eval_mate_threshold:
                return 0.0

        return 1.0 / (1.0 + np.exp(-eval / self.scaling_factor))

    def eval_to_wld(self, eval: int) -> np.ndarray:
        win = self.eval_to_winrate(eval)
        loss = self.eval_to_winrate(-eval)
        draw = max(1.0 - win - loss, 0.0)  # to avoid numerical error
        return np.array([win, loss, draw], dtype=np.float32)
