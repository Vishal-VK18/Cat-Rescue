"""
Cat Rescue OpenEnv Environment
================================
A grid-based reinforcement learning environment where an AI agent
navigates a grid to rescue trapped cats.

Implements the OpenEnv standard interface:
    - reset()  → initialise a fresh episode
    - step()   → apply an action, get (observation, reward, done, info)
    - state()  → return the current grid state

Compatible with:
    - grader.py   (reads observation dict + done/reward values)
    - rewards.py  (can override REWARD_* constants or plug in its own function)
    - server.py   (calls reset / step / state over HTTP)

Grid cell values (also exported as constants so teammates can import them):
    EMPTY    = 0
    WALL     = 1
    CAT      = 2
    AGENT    = 3
    OBSTACLE = 4   (moving obstacle, Level 3 only)
"""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Try to import the openenv-core base class.
# If the library is not installed yet we fall back to a plain object so the
# file is still importable during development / local testing.
# ---------------------------------------------------------------------------
try:
    from openenv.core import BaseEnvironment  # type: ignore
except ImportError:
    class BaseEnvironment:  # type: ignore
        """Minimal stand-in when openenv-core is not installed."""
        pass

# ---------------------------------------------------------------------------
# Cell-type constants  (import these in grader.py / rewards.py)
# ---------------------------------------------------------------------------
EMPTY    = 0
WALL     = 1
CAT      = 2
AGENT    = 3
OBSTACLE = 4

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

# ---------------------------------------------------------------------------
# Reward constants  (rewards.py can override these by monkey-patching or by
# passing a custom rewards dict to CatRescueEnv.__init__)
# ---------------------------------------------------------------------------
REWARD_RESCUE_CAT   =  1.0   # agent steps onto a cat cell
REWARD_HIT_WALL     = -0.3   # agent tries to walk into a wall / boundary
REWARD_STEP         = -0.1   # every step costs a little (encourages speed)
REWARD_ALL_RESCUED  =  2.0   # bonus when every cat on the level is saved

# ---------------------------------------------------------------------------
# Level definitions
# Each level is a dict with:
#   grid_size  : (rows, cols)
#   num_cats   : how many cats are placed
#   walls      : list of (row, col) wall positions  (empty = no static walls)
#   num_moving : number of moving obstacles (Level 3 only)
# ---------------------------------------------------------------------------
LEVELS: Dict[int, Dict[str, Any]] = {
    1: {
        "grid_size": (3, 3),
        "num_cats": 1,
        "walls": [],
        "num_moving": 0,
    },
    2: {
        "grid_size": (5, 5),
        "num_cats": 2,
        # Hand-crafted walls that create simple corridors
        "walls": [(1, 1), (1, 3), (3, 1), (3, 3)],
        "num_moving": 0,
    },
    3: {
        "grid_size": (7, 7),
        "num_cats": 3,
        # A more complex maze-like wall layout
        "walls": [
            (1, 1), (1, 2), (1, 5),
            (2, 4),
            (3, 1), (3, 3), (3, 5),
            (4, 2),
            (5, 1), (5, 4), (5, 5),
        ],
        "num_moving": 2,
    },
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------
class CatRescueEnv(BaseEnvironment):
    """
    Cat Rescue OpenEnv Environment.

    Parameters
    ----------
    level : int
        Difficulty level (1, 2, or 3).
    max_steps : int
        Maximum number of steps per episode before the episode is forced
        to end (prevents infinite loops).
    seed : int | None
        Random seed for reproducibility.
    custom_rewards : dict | None
        Optional dict to override default reward values.
        Keys: 'rescue_cat', 'hit_wall', 'step', 'all_rescued'
    """

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        level: int = 1,
        max_steps: int = 200,
        seed: Optional[int] = None,
        custom_rewards: Optional[Dict[str, float]] = None,
    ):
        if level not in LEVELS:
            raise ValueError(f"Level must be 1, 2, or 3. Got {level}.")

        self.level      = level
        self.max_steps  = max_steps
        self._rng       = random.Random(seed)

        # Load level config
        cfg = LEVELS[level]
        self.rows, self.cols = cfg["grid_size"]
        self._static_walls   = set(map(tuple, cfg["walls"]))
        self._num_cats        = cfg["num_cats"]
        self._num_moving      = cfg["num_moving"]

        # Allow rewards.py to supply its own reward schedule
        self._rewards = {
            "rescue_cat":  REWARD_RESCUE_CAT,
            "hit_wall":    REWARD_HIT_WALL,
            "step":        REWARD_STEP,
            "all_rescued": REWARD_ALL_RESCUED,
        }
        if custom_rewards:
            self._rewards.update(custom_rewards)

        # Internal state (populated by reset())
        self.grid:             List[List[int]] = []
        self.agent_pos:        Tuple[int, int] = (0, 0)
        self.cat_positions:    List[Tuple[int, int]] = []
        self.obstacle_positions: List[Tuple[int, int]] = []
        self._obstacle_dirs:   List[Tuple[int, int]] = []  # velocity per obstacle
        self.cats_rescued:     int = 0
        self.step_count:       int = 0
        self.done:             bool = False

        # Run an initial reset so the env is ready to use immediately
        self.reset()

    # ------------------------------------------------------------------ #
    #  OpenEnv Interface — reset()                                        #
    # ------------------------------------------------------------------ #
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to a fresh starting state.

        Places the agent at (0, 0), randomly distributes cats on empty
        cells, places static walls (Level 2+), and spawns moving
        obstacles (Level 3).

        Returns
        -------
        observation : dict
            The initial observation (same format as step() returns).
        """
        # 1. Build a blank grid filled with EMPTY cells
        self.grid = [[EMPTY] * self.cols for _ in range(self.rows)]

        # 2. Place static walls (Level 2 and 3)
        for (r, c) in self._static_walls:
            self.grid[r][c] = WALL

        # 3. Collect all free cells (not walls) for random placement
        free_cells = self._get_free_cells()

        # 4. Place the agent at (0, 0) — always the top-left corner
        self.agent_pos = (0, 0)
        self.grid[0][0] = AGENT
        free_cells.discard((0, 0))

        # 5. Place cats randomly on remaining free cells
        self.cat_positions = []
        for _ in range(self._num_cats):
            if not free_cells:
                break  # safety guard — shouldn't happen with valid configs
            pos = self._rng.choice(list(free_cells))
            free_cells.discard(pos)
            self.cat_positions.append(pos)
            self.grid[pos[0]][pos[1]] = CAT

        # 6. Place moving obstacles (Level 3 only)
        self.obstacle_positions = []
        self._obstacle_dirs     = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for _ in range(self._num_moving):
            if not free_cells:
                break
            pos = self._rng.choice(list(free_cells))
            free_cells.discard(pos)
            self.obstacle_positions.append(list(pos))   # mutable list
            self._obstacle_dirs.append(self._rng.choice(directions))
            self.grid[pos[0]][pos[1]] = OBSTACLE

        # 7. Reset counters
        self.cats_rescued = 0
        self.step_count   = 0
        self.done         = False

        return self.state()

    # ------------------------------------------------------------------ #
    #  OpenEnv Interface — step()                                         #
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.

        Parameters
        ----------
        action : int
            One of UP (0), DOWN (1), LEFT (2), RIGHT (3).

        Returns
        -------
        observation : dict
            Current state of the environment (from state()).
        reward : float
            Reward received for this step.
        done : bool
            True when the episode has ended (all cats rescued or max_steps hit).
        info : dict
            Extra diagnostic data useful for grader.py and debugging.
            Keys: 'cats_rescued', 'step_count', 'agent_pos',
                  'action_name', 'event'
        """
        if self.done:
            # Episode already finished — return current state with no reward
            return self.state(), 0.0, self.done, self._make_info("episode_done")

        reward = self._rewards["step"]   # base cost for every step
        event  = "step"                  # human-readable event label

        # --- Compute the target cell -----------------------------------
        row, col    = self.agent_pos
        d_row, d_col = self._action_delta(action)
        new_row     = row + d_row
        new_col     = col + d_col

        # --- Check if the move is valid --------------------------------
        if self._is_out_of_bounds(new_row, new_col) or self._is_wall(new_row, new_col):
            # Agent bumps into boundary or wall — penalise and stay put
            reward += self._rewards["hit_wall"]
            event   = "hit_wall"
        else:
            # Valid move — update agent position on the grid
            cell_content = self.grid[new_row][new_col]

            # Clear the agent's old position
            self.grid[row][col] = EMPTY

            # Check what is in the destination cell
            if cell_content == CAT:
                # Cat rescued!
                self.cats_rescued += 1
                self.cat_positions.remove((new_row, new_col))
                reward += self._rewards["rescue_cat"]
                event   = "cat_rescued"

                # Check if ALL cats have been rescued
                if self.cats_rescued == self._num_cats:
                    reward    += self._rewards["all_rescued"]
                    self.done  = True
                    event      = "all_rescued"

            elif cell_content == OBSTACLE:
                # Ran into a moving obstacle — same penalty as a wall
                reward += self._rewards["hit_wall"]
                event   = "hit_obstacle"
                # Agent stays at current position (don't move into obstacle)
                self.grid[row][col] = AGENT
                self.step_count += 1
                return (
                    self.state(),
                    reward,
                    self.done,
                    self._make_info(event, action),
                )

            # Move agent to new position
            self.agent_pos = (new_row, new_col)
            self.grid[new_row][new_col] = AGENT

        # --- Move obstacles (Level 3) ----------------------------------
        if self._num_moving > 0:
            self._move_obstacles()

        # --- Check step limit ------------------------------------------
        self.step_count += 1
        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            event     = "max_steps_reached"

        info = self._make_info(event, action)
        return self.state(), reward, self.done, info

    # ------------------------------------------------------------------ #
    #  OpenEnv Interface — state()                                        #
    # ------------------------------------------------------------------ #
    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state as an observation dict.

        This is what grader.py reads to evaluate the agent.

        Returns
        -------
        dict with keys:
            'grid'             : 2-D list of int (deep copy — safe to mutate)
            'agent_pos'        : (row, col) tuple
            'cat_positions'    : list of (row, col) tuples
            'cats_rescued'     : int — how many cats saved so far
            'cats_remaining'   : int — how many cats still on the grid
            'step_count'       : int
            'done'             : bool
            'level'            : int
            'grid_size'        : (rows, cols)
        """
        return {
            "grid":           copy.deepcopy(self.grid),
            "agent_pos":      self.agent_pos,
            "cat_positions":  list(self.cat_positions),
            "cats_rescued":   self.cats_rescued,
            "cats_remaining": self._num_cats - self.cats_rescued,
            "step_count":     self.step_count,
            "done":           self.done,
            "level":          self.level,
            "grid_size":      (self.rows, self.cols),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    def _action_delta(self, action: int) -> Tuple[int, int]:
        """Map an action integer to a (row_delta, col_delta) pair."""
        return {
            UP:    (-1,  0),
            DOWN:  ( 1,  0),
            LEFT:  ( 0, -1),
            RIGHT: ( 0,  1),
        }.get(action, (0, 0))   # (0, 0) → no-op for unknown actions

    def _is_out_of_bounds(self, row: int, col: int) -> bool:
        """Return True if (row, col) is outside the grid."""
        return not (0 <= row < self.rows and 0 <= col < self.cols)

    def _is_wall(self, row: int, col: int) -> bool:
        """Return True if the cell at (row, col) contains a static wall."""
        return self.grid[row][col] == WALL

    def _get_free_cells(self) -> set:
        """Return a set of (row, col) positions that are currently EMPTY."""
        free = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == EMPTY:
                    free.add((r, c))
        return free

    def _move_obstacles(self) -> None:
        """
        Move each obstacle one step in its current direction.
        If the next cell is a wall, boundary, or another obstacle the
        obstacle bounces — its direction is reversed.

        The agent is never overwritten: obstacles bounce off the agent
        cell too, keeping collision detection clean.
        """
        for i, (obs_pos, obs_dir) in enumerate(
            zip(self.obstacle_positions, self._obstacle_dirs)
        ):
            r, c   = obs_pos
            dr, dc = obs_dir
            nr, nc = r + dr, c + dc

            # Clear current obstacle cell (only if it hasn't been overwritten
            # by the agent moving there in the same step)
            if self.grid[r][c] == OBSTACLE:
                self.grid[r][c] = EMPTY

            # Decide whether to bounce or advance
            blocked = (
                self._is_out_of_bounds(nr, nc)
                or self.grid[nr][nc] in (WALL, OBSTACLE, AGENT)
            )

            if blocked:
                # Reverse direction (bounce)
                self._obstacle_dirs[i] = (-dr, -dc)
                nr, nc = r, c   # stay in place this tick

            self.obstacle_positions[i] = [nr, nc]
            self.grid[nr][nc] = OBSTACLE

    def _make_info(self, event: str, action: Optional[int] = None) -> Dict[str, Any]:
        """Build the info dict returned alongside every step."""
        return {
            "cats_rescued": self.cats_rescued,
            "step_count":   self.step_count,
            "agent_pos":    self.agent_pos,
            "action_name":  ACTION_NAMES.get(action, "N/A") if action is not None else "N/A",
            "event":        event,
        }

    # ------------------------------------------------------------------ #
    #  Convenience / debug helpers (not part of the OpenEnv interface)    #
    # ------------------------------------------------------------------ #
    def render(self) -> str:
        """
        Return a human-readable ASCII string of the current grid.
        Useful for debugging and demos.

        Symbols:
            .  EMPTY
            #  WALL
            C  CAT
            A  AGENT
            X  MOVING OBSTACLE
        """
        symbols = {EMPTY: ".", WALL: "#", CAT: "C", AGENT: "A", OBSTACLE: "X"}
        lines = []
        for row in self.grid:
            lines.append(" ".join(symbols.get(cell, "?") for cell in row))
        header = f"Level {self.level} | Step {self.step_count} | Cats rescued {self.cats_rescued}/{self._num_cats}"
        return header + "\n" + "\n".join(lines)

    def action_space(self) -> List[int]:
        """Return the list of valid action integers."""
        return [UP, DOWN, LEFT, RIGHT]

    def observation_space(self) -> Dict[str, Any]:
        """
        Describe the observation space for grader.py / server.py.
        Returns a metadata dict (not a sampled observation).
        """
        return {
            "grid_shape":    (self.rows, self.cols),
            "cell_types":    {"EMPTY": EMPTY, "WALL": WALL, "CAT": CAT,
                              "AGENT": AGENT, "OBSTACLE": OBSTACLE},
            "agent_pos":     "Tuple[int, int]",
            "cat_positions": "List[Tuple[int, int]]",
            "cats_rescued":  "int",
            "step_count":    "int",
            "done":          "bool",
        }


# ---------------------------------------------------------------------------
# Quick smoke-test  —  run `python environment.py` to verify everything works
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for lvl in (1, 2, 3):
        print(f"\n{'='*40}")
        print(f"  Smoke-test — Level {lvl}")
        print(f"{'='*40}")

        env = CatRescueEnv(level=lvl, seed=42)
        print(env.render())
        print()

        # Take 10 random steps
        for _ in range(10):
            action = random.choice(env.action_space())
            obs, reward, done, info = env.step(action)
            print(f"Action: {ACTION_NAMES[action]:<5} | "
                  f"Reward: {reward:+.1f} | "
                  f"Event: {info['event']:<16} | "
                  f"Agent: {info['agent_pos']} | "
                  f"Rescued: {info['cats_rescued']}")
            if done:
                print("  >> Episode finished early!")
                break

        print()
        print(env.render())
