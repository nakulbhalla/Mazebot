import numpy as np
import gymnasium as gym
from collections import deque

"""
Improved custom.py for CISC 474 Coverage Tournament
- 9-channel observation space (adds danger gradient + coverage ratio channels)
- Tuned reward function for PPO with cleaner signal separation
- Designed for Stable Baselines3 PPO
"""

# ────────────────────────────────────────────────────────
#  Reward memory (global state, reset each episode)
# ────────────────────────────────────────────────────────
_LAST_OBS = None
_PREV_POS = None
_PREV_PREV_POS = None
_PREV_FRONTIER_DIST = None
_PREV_DANGER_DIST = None
_PREV_ENEMY_DIST = None
_PREV_STEPS_REMAINING = None
_RECENT_POSITIONS = deque(maxlen=8)


def _reset_reward_memory():
    global _LAST_OBS, _PREV_POS, _PREV_PREV_POS
    global _PREV_FRONTIER_DIST, _PREV_DANGER_DIST, _PREV_ENEMY_DIST
    global _PREV_STEPS_REMAINING, _RECENT_POSITIONS

    _LAST_OBS = None
    _PREV_POS = None
    _PREV_PREV_POS = None
    _PREV_FRONTIER_DIST = None
    _PREV_DANGER_DIST = None
    _PREV_ENEMY_DIST = None
    _PREV_STEPS_REMAINING = None
    _RECENT_POSITIONS = deque(maxlen=8)


# ────────────────────────────────────────────────────────
#  RGB colour definitions (from env.py, copied to avoid import)
# ────────────────────────────────────────────────────────
BLACK     = np.array([0, 0, 0],       dtype=np.uint8)   # unexplored
WHITE     = np.array([255, 255, 255], dtype=np.uint8)   # explored
BROWN     = np.array([101, 67, 33],   dtype=np.uint8)   # wall
GREY      = np.array([160, 161, 161], dtype=np.uint8)   # agent
GREEN     = np.array([31, 198, 0],    dtype=np.uint8)   # enemy
RED       = np.array([255, 0, 0],     dtype=np.uint8)   # unexplored + danger
LIGHT_RED = np.array([255, 127, 127], dtype=np.uint8)   # explored + danger


# ────────────────────────────────────────────────────────
#  Helper utilities
# ────────────────────────────────────────────────────────
def _color_mask(grid: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Boolean mask for cells whose RGB matches `color`."""
    return np.all(grid == color, axis=2)


def _frontier_mask(unexplored_mask: np.ndarray, explored_mask: np.ndarray) -> np.ndarray:
    """
    Frontier = unexplored cells adjacent (4-neighbourhood) to at least one explored cell.
    """
    adj = np.zeros_like(explored_mask, dtype=bool)
    adj[1:, :]  |= explored_mask[:-1, :]
    adj[:-1, :] |= explored_mask[1:, :]
    adj[:, 1:]  |= explored_mask[:, :-1]
    adj[:, :-1] |= explored_mask[:, 1:]
    return unexplored_mask & adj


def _min_manhattan_distance(mask: np.ndarray, row: int, col: int):
    """
    Minimum Manhattan distance from (row, col) to any True cell in mask.
    Returns None if mask has no True cells.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    dists = np.abs(coords[:, 0] - row) + np.abs(coords[:, 1] - col)
    return int(dists.min())


def _danger_gradient(danger_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Soft threat map: each cell gets value 1/(1 + dist_to_nearest_danger).
    Gives PPO smooth gradient signal instead of a hard binary mask.
    Cells inside danger zones get value 1.0.
    """
    coords = np.argwhere(danger_mask)
    if coords.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    rows = np.arange(h).reshape(-1, 1)
    cols = np.arange(w).reshape(1, -1)

    # Vectorised: min Manhattan dist for every cell
    min_dist = np.full((h, w), np.inf)
    for r, c in coords:
        d = np.abs(rows - r) + np.abs(cols - c)
        np.minimum(min_dist, d, out=min_dist)

    gradient = 1.0 / (1.0 + min_dist)
    return gradient.astype(np.float32)


# ────────────────────────────────────────────────────────
#  OBSERVATION SPACE  (9 channels, 9 × H × W)
# ────────────────────────────────────────────────────────
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    9-channel float32 observation:
      0: walls
      1: unexplored cells
      2: explored cells
      3: danger FOV (binary)
      4: agent position
      5: enemy positions
      6: frontier cells
      7: danger gradient (soft distance-weighted threat map)   ← NEW
      8: global coverage ratio broadcast to all cells          ← NEW
    """
    h, w = env.grid.shape[:2]
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(9, h, w),
        dtype=np.float32
    )


def observation(grid: np.ndarray) -> np.ndarray:
    """
    Convert the RGB grid into a 9-channel binary/float tensor.
    """
    h, w = grid.shape[:2]

    # Colour masks
    wall_mask      = _color_mask(grid, BROWN)
    black_mask     = _color_mask(grid, BLACK)
    white_mask     = _color_mask(grid, WHITE)
    grey_mask      = _color_mask(grid, GREY)
    green_mask     = _color_mask(grid, GREEN)
    red_mask       = _color_mask(grid, RED)
    light_red_mask = _color_mask(grid, LIGHT_RED)

    # Semantic channels
    walls     = wall_mask
    unexplored = black_mask | red_mask
    explored  = white_mask | light_red_mask | grey_mask
    danger    = red_mask | light_red_mask
    agent     = grey_mask
    enemies   = green_mask
    frontier  = _frontier_mask(unexplored, explored)

    # Ch 7: danger gradient (soft threat map)
    danger_grad = _danger_gradient(danger, h, w)

    # Ch 8: global coverage ratio broadcast to all cells
    total_explored = int(explored.sum())
    total_cells    = h * w - int(walls.sum())
    coverage_ratio = total_explored / max(1, total_cells)
    coverage_ch    = np.full((h, w), coverage_ratio, dtype=np.float32)

    obs = np.stack(
        [
            walls.astype(np.float32),       # 0
            unexplored.astype(np.float32),  # 1
            explored.astype(np.float32),    # 2
            danger.astype(np.float32),      # 3
            agent.astype(np.float32),       # 4
            enemies.astype(np.float32),     # 5
            frontier.astype(np.float32),    # 6
            danger_grad,                    # 7  ← NEW
            coverage_ch,                    # 8  ← NEW
        ],
        axis=0
    )

    global _LAST_OBS
    _LAST_OBS = obs
    return obs


# ────────────────────────────────────────────────────────
#  REWARD FUNCTION
# ────────────────────────────────────────────────────────
def reward(info: dict) -> float:
    """
    Reward design principles:
    - Exploration is the primary objective → strongest positive signals
    - Safety (avoiding FOV) is a hard constraint → strong negative signals
    - Frontier shaping gives directional guidance without overpowering coverage
    - Anti-loop penalties are aggressive to prevent repetitive behaviour
    - Terminal rewards dominate, providing clear episode-level signal
    """
    global _PREV_POS, _PREV_PREV_POS
    global _PREV_FRONTIER_DIST, _PREV_DANGER_DIST, _PREV_ENEMY_DIST
    global _PREV_STEPS_REMAINING, _RECENT_POSITIONS

    agent_pos           = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining     = info["cells_remaining"]
    coverable_cells     = info["coverable_cells"]
    steps_remaining     = info["steps_remaining"]
    new_cell_covered    = info["new_cell_covered"]
    game_over           = info["game_over"]

    # ── Detect new episode ──────────────────────────────
    if _PREV_STEPS_REMAINING is None or steps_remaining > _PREV_STEPS_REMAINING:
        _PREV_POS            = None
        _PREV_PREV_POS       = None
        _PREV_FRONTIER_DIST  = None
        _PREV_DANGER_DIST    = None
        _PREV_ENEMY_DIST     = None
        _RECENT_POSITIONS    = deque(maxlen=8)

    row = agent_pos // 10
    col = agent_pos % 10

    covered_ratio = total_covered_cells / max(1, coverable_cells)
    r = 0.0

    # ── 1) Step cost ────────────────────────────────────
    # Small constant cost keeps episodes tight.
    r -= 0.01

    # ── 2) Exploration reward ───────────────────────────
    if new_cell_covered:
        # Base reward + late-game bonus (harder to explore last cells)
        # Ramps up from 1.0 to 1.5 as coverage increases.
        r += 1.0 + 0.5 * covered_ratio
    else:
        if _PREV_POS is not None and agent_pos == _PREV_POS:
            # STAY / bumped into wall / blocked — strong penalty
            r -= 0.20
        else:
            # Moved but over already-explored ground
            r -= 0.08

    # ── 3) Frontier shaping ─────────────────────────────
    # Reward moving toward unexplored edges.
    # Uses Ch 6 from observation.
    if _LAST_OBS is not None:
        frontier_mask = _LAST_OBS[6] > 0.5
        frontier_dist = _min_manhattan_distance(frontier_mask, row, col)

        if frontier_dist is not None:
            if _PREV_FRONTIER_DIST is not None:
                delta = _PREV_FRONTIER_DIST - frontier_dist  # positive = closer
                r += 0.07 * np.clip(delta, -2, 2)

            # Adjacency bonus: standing next to a frontier cell
            if frontier_dist <= 1:
                r += 0.05

            _PREV_FRONTIER_DIST = frontier_dist
        else:
            _PREV_FRONTIER_DIST = None

    # ── 4) Safety shaping ───────────────────────────────
    # Penalise proximity to danger zones and enemies.
    # Uses the soft danger gradient (Ch 7) for smoother signal,
    # plus binary distance checks for hard penalties.
    if _LAST_OBS is not None:
        # Agent's own danger gradient value (how threatened is current cell)
        agent_threat = float(_LAST_OBS[7, row, col])
        # Penalise proportionally to threat level — always active
        r -= 0.25 * agent_threat

        danger_mask = _LAST_OBS[3] > 0.5
        enemy_mask  = _LAST_OBS[5] > 0.5

        danger_dist = _min_manhattan_distance(danger_mask, row, col)
        enemy_dist  = _min_manhattan_distance(enemy_mask,  row, col)

        # Hard distance penalties (danger FOV)
        if danger_dist is not None:
            if danger_dist == 0:    # inside FOV
                r -= 0.40
            elif danger_dist == 1:
                r -= 0.20
            elif danger_dist == 2:
                r -= 0.08

            # Shaping: reward moving away from danger
            if _PREV_DANGER_DIST is not None:
                delta = danger_dist - _PREV_DANGER_DIST  # positive = farther
                r += 0.04 * np.clip(delta, -2, 2)

            _PREV_DANGER_DIST = danger_dist
        else:
            _PREV_DANGER_DIST = None

        # Hard distance penalties (enemies themselves)
        if enemy_dist is not None:
            if enemy_dist <= 1:
                r -= 0.20
            elif enemy_dist == 2:
                r -= 0.08

            _PREV_ENEMY_DIST = enemy_dist
        else:
            _PREV_ENEMY_DIST = None

    # ── 5) Anti-loop / oscillation penalties ────────────
    # A-B-A oscillation
    if (_PREV_PREV_POS is not None
            and agent_pos == _PREV_PREV_POS
            and not new_cell_covered):
        r -= 0.15

    # Revisiting recent positions
    revisit_count = sum(1 for p in _RECENT_POSITIONS if p == agent_pos)
    if revisit_count > 0 and not new_cell_covered:
        # Scales with how many times we've been here recently
        r -= 0.06 * min(revisit_count, 3)

    # ── 6) Terminal rewards ─────────────────────────────
    if cells_remaining == 0:
        # Full coverage! Bonus for finishing early.
        speed_bonus = 3.0 * (steps_remaining / 500.0)
        r += 15.0 + speed_bonus

    elif game_over:
        # Caught by enemy FOV — large penalty, scaled by how incomplete we are.
        # This makes the agent care more about safety when it's made progress.
        incomplete_penalty = 4.0 * (1.0 - covered_ratio)
        r -= 15.0 + incomplete_penalty

    elif steps_remaining <= 0 and cells_remaining > 0:
        # Timeout: penalty proportional to remaining unexplored area
        r -= 5.0 + 5.0 * (1.0 - covered_ratio)

    # ── Update memory ───────────────────────────────────
    _RECENT_POSITIONS.append(agent_pos)
    _PREV_PREV_POS       = _PREV_POS
    _PREV_POS            = agent_pos
    _PREV_STEPS_REMAINING = steps_remaining

    return float(r)


# ────────────────────────────────────────────────────────
#  RECOMMENDED PPO HYPERPARAMETERS  (Stable Baselines3)
# ────────────────────────────────────────────────────────
"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

# Build vectorised env (8 parallel workers)
def make_env():
    def _init():
        env = YourEnv()   # replace with your env constructor
        return env
    return _init

n_envs = 8
env = VecMonitor(SubprocVecEnv([make_env() for _ in range(n_envs)]))

model = PPO(
    policy            = "MlpPolicy",   # or CnnPolicy if obs is image-like
    env               = env,

    # ── Core hyperparameters ──────────────────────────────
    n_steps           = 512,           # rollout per env before update
    batch_size        = 256,           # minibatch for gradient update
    n_epochs          = 10,            # passes over each rollout buffer
    gamma             = 0.995,         # high discount — rewards 500-step episodes
    gae_lambda        = 0.95,          # GAE lambda (bias/variance trade-off)
    clip_range        = 0.2,           # PPO clip epsilon
    clip_range_vf     = None,          # don't clip value loss (usually fine)
    ent_coef          = 0.01,          # entropy bonus — encourages exploration
    vf_coef           = 0.5,           # value function loss weight
    max_grad_norm     = 0.5,           # gradient clipping

    # ── Learning rate ─────────────────────────────────────
    # Linear decay from 3e-4 → 0 helps convergence in later training.
    learning_rate     = lambda progress: 3e-4 * (1.0 - progress),

    # ── Policy network ────────────────────────────────────
    policy_kwargs = dict(
        net_arch = dict(
            pi = [256, 256, 128],      # actor: 3 layers, tapers down
            vf = [256, 256, 128],      # critic: same
        ),
        activation_fn = __import__("torch").nn.Tanh,
    ),

    verbose           = 1,
    tensorboard_log   = "./tb_logs/",
)

# ── Callbacks ─────────────────────────────────────────────
eval_callback = EvalCallback(
    eval_env          = VecMonitor(SubprocVecEnv([make_env()])),
    best_model_save_path = "./best_model/",
    eval_freq         = 20_000,
    n_eval_episodes   = 20,
    deterministic     = True,
    verbose           = 1,
)

# ── Training ──────────────────────────────────────────────
model.learn(
    total_timesteps   = 5_000_000,     # 5M steps; increase to 10M for best results
    callback          = eval_callback,
    tb_log_name       = "PPO_coverage",
)
model.save("coverage_agent_final")

# ── Curriculum / training strategy tips ───────────────────
# 1. Start with 1 enemy, train to 80%+ coverage, then add more enemies.
# 2. Randomise enemy start positions every episode to prevent overfitting to one map.
# 3. Generate random wall configs (with guaranteed connectivity) for robustness.
# 4. Use VecNormalize to normalise observations and rewards if training is unstable:
#    from stable_baselines3.common.vec_env import VecNormalize
#    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
"""
