from typing import Optional

import optuna
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from common.plot import visualize


class TrailEvalCallback(EvalCallback):

    def __init__(
            self, 
            eval_env: VecEnv,
            trail: optuna.Trial,
            n_eval_episodes: int = 5, 
            eval_freq: int = 10000, 
            deterministic: bool = True,
            log_path: Optional[str] = None, 
            best_model_save_path: Optional[str] = None,
            verbose: int = 1
            ):
        
        super(TrailEvalCallback, self).__init__(
            eval_env = eval_env, 
            n_eval_episodes = n_eval_episodes,
            eval_freq = eval_freq,
            deterministic = deterministic, 
            verbose = verbose,
            best_model_save_path = best_model_save_path,
            log_path = log_path,
            )
        
        self.trail = trail
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrailEvalCallback, self)._on_step()
            self.eval_idx += 1
            self.trail.report(self.last_mean_reward, step=self.eval_idx)
            if self.trail.should_prune():
                self.is_pruned = True
                return False
        return True
    
class PlotCallback(BaseCallback):
    
    def __init__(self, verbose: int = 0, logdir: str = None):
        self.logdir = logdir
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        if self.num_timesteps % 2560 == 0:
            visualize(self.logdir)
            
        return super()._on_rollout_end()
