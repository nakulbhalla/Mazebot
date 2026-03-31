import csv
from pathlib import Path
from typing import Dict, List

import gymnasium
import coverage_gridworld
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecMonitor


# =============================================================================
# CNN FEATURE EXTRACTOR
# =============================================================================
class SmallGridCNN(BaseFeaturesExtractor):
    """
    CNN for the 9 x H x W observation from custom.py.

    If you change the number of observation channels in custom.py,
    this class automatically adapts because it reads observation_space.shape[0].
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    normalize_images=False,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
MAX_STEPS_PER_EPISODE = 500
MODELS_DIR = Path("./models")
LOGS_DIR = Path("./logs")
CHECKPOINTS_DIR = Path("./checkpoints")
CSV_DIR = Path("./eval_csv")

for directory in [MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, CSV_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def linear_schedule(initial_value: float):
    """
    Correct SB3 linear schedule.

    In Stable Baselines3, progress_remaining starts at 1.0 and goes to 0.0.
    So lr should be: initial_value * progress_remaining
    """
    return lambda progress_remaining: initial_value * progress_remaining


def make_train_env(env_id: str, n_envs: int = 8) -> VecMonitor:
    """
    Vectorized training environment.

    Change n_envs here if you want more/fewer parallel workers.
    """
    return VecMonitor(make_vec_env(env_id, n_envs=n_envs))


# =============================================================================
# CUSTOM EVALUATION CALLBACK
# Logs:
# - reward
# - episode length
# - coverage ratio
# - success rate
# - game-over rate
# - timeout rate
# - steps used
# =============================================================================
class CoverageEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env_id: str,
        stage_name: str,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 20,
        deterministic: bool = True,
        best_model_dir: str = "./models",
        csv_path: str | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_id = eval_env_id
        self.stage_name = stage_name
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_dir = Path(best_model_dir)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.best_mean_coverage = -float("inf")
        self.best_mean_steps = float("inf")
        self.last_eval_step = 0
        self.csv_path = Path(csv_path) if csv_path is not None else None

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timesteps",
                    "mean_reward",
                    "mean_ep_length",
                    "mean_coverage",
                    "success_rate",
                    "game_over_rate",
                    "timeout_rate",
                    "mean_steps_used",
                ])

    def _evaluate_once(self) -> Dict[str, float]:
        env = gymnasium.make(
            self.eval_env_id,
            render_mode=None,
            predefined_map_list=None,
            activate_game_status=False,
        )

        rewards: List[float] = []
        ep_lengths: List[int] = []
        coverage_ratios: List[float] = []
        success_flags: List[int] = []
        game_over_flags: List[int] = []
        timeout_flags: List[int] = []
        steps_used_list: List[int] = []

        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            episode_steps = 0
            final_info = None

            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                episode_steps += 1
                final_info = info

            if final_info is None:
                continue

            total_covered_cells = final_info["total_covered_cells"]
            coverable_cells = final_info["coverable_cells"]
            cells_remaining = final_info["cells_remaining"]
            steps_remaining = final_info["steps_remaining"]
            game_over = bool(final_info["game_over"])

            coverage_ratio = total_covered_cells / max(1, coverable_cells)
            success = int(cells_remaining == 0 and not game_over)
            timeout = int(steps_remaining <= 0 and cells_remaining > 0 and not game_over)
            steps_used = MAX_STEPS_PER_EPISODE - steps_remaining

            rewards.append(total_reward)
            ep_lengths.append(episode_steps)
            coverage_ratios.append(coverage_ratio)
            success_flags.append(success)
            game_over_flags.append(int(game_over))
            timeout_flags.append(timeout)
            steps_used_list.append(steps_used)

        env.close()

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_ep_length": float(np.mean(ep_lengths)),
            "mean_coverage": float(np.mean(coverage_ratios)),
            "success_rate": float(np.mean(success_flags)),
            "game_over_rate": float(np.mean(game_over_flags)),
            "timeout_rate": float(np.mean(timeout_flags)),
            "mean_steps_used": float(np.mean(steps_used_list)),
        }

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval_step) < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps
        metrics = self._evaluate_once()

        # Log into TensorBoard / SB3 logger
        self.logger.record("eval/mean_reward", metrics["mean_reward"])
        self.logger.record("eval/mean_ep_length", metrics["mean_ep_length"])
        self.logger.record("eval_custom/mean_coverage", metrics["mean_coverage"])
        self.logger.record("eval_custom/success_rate", metrics["success_rate"])
        self.logger.record("eval_custom/game_over_rate", metrics["game_over_rate"])
        self.logger.record("eval_custom/timeout_rate", metrics["timeout_rate"])
        self.logger.record("eval_custom/mean_steps_used", metrics["mean_steps_used"])
        self.logger.dump(self.num_timesteps)

        # Save CSV row for later analysis
        if self.csv_path is not None:
            with self.csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    metrics["mean_reward"],
                    metrics["mean_ep_length"],
                    metrics["mean_coverage"],
                    metrics["success_rate"],
                    metrics["game_over_rate"],
                    metrics["timeout_rate"],
                    metrics["mean_steps_used"],
                ])

        # Best model rule:
        # 1) higher coverage is better
        # 2) if tied, fewer steps used is better
        improved = False
        if metrics["mean_coverage"] > self.best_mean_coverage:
            improved = True
        elif np.isclose(metrics["mean_coverage"], self.best_mean_coverage):
            if metrics["mean_steps_used"] < self.best_mean_steps:
                improved = True

        if improved:
            self.best_mean_coverage = metrics["mean_coverage"]
            self.best_mean_steps = metrics["mean_steps_used"]
            self.model.save(self.best_model_dir / f"{self.stage_name}_best")

        if self.verbose:
            print("\n" + "-" * 70)
            print(f"EVAL -- {self.stage_name} -- step {self.num_timesteps:,}")
            print(f"mean_reward     : {metrics['mean_reward']:.3f}")
            print(f"mean_ep_length  : {metrics['mean_ep_length']:.2f}")
            print(f"mean_coverage   : {metrics['mean_coverage']:.3f}")
            print(f"success_rate    : {metrics['success_rate']:.3f}")
            print(f"game_over_rate  : {metrics['game_over_rate']:.3f}")
            print(f"timeout_rate    : {metrics['timeout_rate']:.3f}")
            print(f"mean_steps_used : {metrics['mean_steps_used']:.2f}")
            print("-" * 70)

        return True


# =============================================================================
# CURRICULUM
# No mixed-map stage included.
# =============================================================================
CURRICULUM = [
    {
        "env_id": "just_go",
        "timesteps": 500_000,
        "ent_coef": 0.020,
        "lr": 3e-4,
        "save_name": "stage1_just_go",
    },
    {
        "env_id": "safe",
        "timesteps": 750_000,
        "ent_coef": 0.015,
        "lr": 2e-4,
        "save_name": "stage2_safe",
    },
    {
        "env_id": "maze",
        "timesteps": 1_000_000,
        "ent_coef": 0.012,
        "lr": 1.5e-4,
        "save_name": "stage3_maze",
    },
    {
        "env_id": "chokepoint",
        "timesteps": 1_000_000,
        "ent_coef": 0.008,
        "lr": 7.5e-5,
        "save_name": "stage4_chokepoint",
    },
    {
        "env_id": "sneaky_enemies",
        "timesteps": 2_000_000,
        "ent_coef": 0.005,
        "lr": 5e-5,
        "save_name": "stage5_sneaky_enemies",
    },
]


# =============================================================================
# TRAINING LOOP
# =============================================================================
model = None

for stage_idx, stage in enumerate(CURRICULUM, start=1):
    env_id = stage["env_id"]
    timesteps = stage["timesteps"]
    ent_coef = stage["ent_coef"]
    lr = stage["lr"]
    save_name = stage["save_name"]

    print("\n" + "=" * 70)
    print(f"STAGE {stage_idx}/{len(CURRICULUM)} -- {env_id.upper()}")
    print(f"timesteps : {timesteps:,}")
    print(f"ent_coef  : {ent_coef}")
    print(f"lr start  : {lr}")
    print("=" * 70)

    train_env = make_train_env(env_id, n_envs=8)

    eval_callback = CoverageEvalCallback(
        eval_env_id=env_id,
        stage_name=save_name,
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        best_model_dir=str(MODELS_DIR / f"{save_name}_best"),
        csv_path=str(CSV_DIR / f"{save_name}_eval.csv"),
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINTS_DIR / save_name),
        name_prefix="ckpt",
        verbose=0,
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    if model is None:
        model = PPO(
            "CnnPolicy",
            train_env,
            device="cuda",
            policy_kwargs=policy_kwargs,
            learning_rate=linear_schedule(lr),
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./tb_logs/",
            verbose=1,
        )
    else:
        # Keep learned weights, but swap to the new stage environment.
        model.set_env(train_env)
        # IMPORTANT: update BOTH the scalar attribute and the actual SB3 schedule.
        model.ent_coef = ent_coef
        model.learning_rate = linear_schedule(lr)
        model.lr_schedule = linear_schedule(lr)

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=save_name,
        # Reset stage counters so each stage gets its own clean LR schedule and logs.
        reset_num_timesteps=True,
    )

    model.save(MODELS_DIR / save_name)
    print(f"Stage {stage_idx} complete -- saved to '{MODELS_DIR / save_name}'.")

    train_env.close()

print("\nAll stages complete.")
print(f"Final model saved as: {MODELS_DIR / 'stage5_sneaky_enemies'}")


# =============================================================================
# FINAL VISUAL EVALUATION
# Change env_id here if you want to watch a different stage map.
# =============================================================================
final_demo_env_id = "sneaky_enemies"

env = gymnasium.make(
    final_demo_env_id,
    render_mode="human",
    predefined_map_list=None,
    activate_game_status=True,
)

model = PPO.load(MODELS_DIR / "stage5_sneaky_enemies", env=env)

obs, _ = env.reset()
terminated = False
truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        "action:", action,
        "| reward:", round(float(reward), 3),
        "| new_cell:", info["new_cell_covered"],
        "| remaining:", info["cells_remaining"],
        "| agent_pos:", info["agent_pos"],
    )

    total_reward += float(reward)

print(f"\nTotal reward: {round(total_reward, 3)}")
env.close()
