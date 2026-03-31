from stable_baselines3 import PPO
import gymnasium
import coverage_gridworld
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments for training
vec_env = make_vec_env("just_go", n_envs=4)

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1
)

model.learn(total_timesteps=10000)
model.save("ppo_gridworld")

# Single environment for testing
env = gymnasium.make(
    "just_go",
    render_mode="human",
    predefined_map_list=None,
    activate_game_status=True
)

model = PPO.load("ppo_gridworld")

obs, _ = env.reset()
terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
env.close()