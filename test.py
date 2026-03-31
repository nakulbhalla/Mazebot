from stable_baselines3 import PPO
import gymnasium
import coverage_gridworld
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

"""
class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 7

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False,
)

# -------- TRAIN --------
vec_env = make_vec_env("sneaky enemies", n_envs=8)

model = PPO(
    "CnnPolicy",
    vec_env,
    device="cuda",
    policy_kwargs=policy_kwargs,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1
)

model.learn(total_timesteps=200000)
model.save("ppo_gridworld_cnn")
"""

# -------- TEST --------
env = gymnasium.make(
    "just_go",
    render_mode="human",
    predefined_map_list=None,
    activate_game_status=True
)

model = PPO.load("stage1_just_go", env=env)

obs, _ = env.reset()
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        "action:", action,
        "| reward:", reward,
        "| new_cell:", info["new_cell_covered"],
        "| remaining:", info["cells_remaining"],
        "| agent_pos:", info["agent_pos"]
    )

    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()