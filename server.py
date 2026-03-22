"""
Cat Rescue — OpenEnv FastAPI Server
=====================================
Wraps CatRescueEnv, CatRescueGrader, and CatRescueRewards into a
standard OpenEnv HTTP API.

Endpoints
---------
POST /reset          → start a new episode (calls env.reset())
POST /step           → take one action    (calls env.step(action))
GET  /state          → read current state (calls env.state())
POST /grade          → score an episode   (calls grader.grade(episode_log))

Run locally
-----------
    uvicorn server:app --host 0.0.0.0 --port 7860

HuggingFace Spaces expects port 7860, which is why we default to it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Import the three core modules written by the team
# ---------------------------------------------------------------------------
from environment import CatRescueEnv           # YOUR file
from grader import CatRescueGrader             # teammate's file
from rewards import CatRescueRewards           # teammate's file

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cat Rescue OpenEnv",
    description=(
        "Grid-based AI environment for the Meta × PyTorch OpenEnv Hackathon. "
        "An agent navigates a grid to rescue trapped cats."
    ),
    version="1.0.0",
)

# Allow cross-origin requests (needed for HF Spaces iframes / web agents)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared singleton instances
# A single environment instance is kept alive between requests so that
# the agent can call /reset once and then repeatedly call /step.
# ---------------------------------------------------------------------------
env     = CatRescueEnv(level=1)          # default level; /reset can change it
grader  = CatRescueGrader()
rewards = CatRescueRewards()


# ---------------------------------------------------------------------------
# Request / Response schemas  (Pydantic keeps the API self-documenting)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    level: int = Field(default=1, ge=1, le=3, description="Difficulty level (1, 2, or 3).")
    max_steps: int = Field(default=200, ge=1, description="Max steps before episode ends.")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")


class StepRequest(BaseModel):
    """Body for POST /step."""
    action: int = Field(
        ...,
        ge=0,
        le=3,
        description="Action to take. 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.",
    )


class StepResponse(BaseModel):
    """Response from POST /step."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeRequest(BaseModel):
    """Body for POST /grade.  episode_log is a list of step records."""
    episode_log: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Ordered list of step dicts, each containing at least "
            "'action', 'reward', 'done', and 'info' keys."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["UI"])
def root() -> FileResponse:
    """
    Serve the game UI.
    Returns index.html so players can open the root URL in a browser.
    """
    return FileResponse("index.html", media_type="text/html")


@app.post("/reset", tags=["OpenEnv"])
def reset(body: ResetRequest) -> Dict[str, Any]:
    """
    **POST /reset** — Start a fresh episode.

    Re-initialises the environment with the requested level, max_steps,
    and optional seed, then returns the initial observation.

    This follows the OpenEnv standard: always call /reset before /step.
    """
    global env  # replace the singleton with a freshly configured instance

    env = CatRescueEnv(
        level=body.level,
        max_steps=body.max_steps,
        seed=body.seed,
    )

    # env.reset() is already called inside __init__; call again for clarity
    # and to return the canonical initial observation.
    observation = env.reset()
    return {"observation": observation}


@app.post("/step", tags=["OpenEnv"], response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    """
    **POST /step** — Execute one action.

    The agent POSTs `{"action": <int>}` and receives back:
    - `observation` — the new grid state
    - `reward`      — reward for this transition
    - `done`        — whether the episode has ended
    - `info`        — diagnostic dict (event, agent_pos, cats_rescued …)

    Raises 400 if the episode is already finished (call /reset first).
    """
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. Call POST /reset to start a new one.",
        )

    observation, reward, done, info = env.step(body.action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", tags=["OpenEnv"])
def state() -> Dict[str, Any]:
    """
    **GET /state** — Read the current environment state without advancing it.

    Returns the same observation dict that /step returns, but does NOT
    consume a step or change any game state.
    """
    return {"observation": env.state()}


@app.post("/grade", tags=["Grading"])
def grade(body: GradeRequest) -> Dict[str, Any]:
    """
    **POST /grade** — Score a completed episode.

    Accepts the full `episode_log` (list of step records collected by
    the agent) and delegates scoring to CatRescueGrader.

    The grader returns a score dict; exact keys depend on grader.py.
    """
    try:
        result = grader.grade(body.episode_log)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grader error: {exc}") from exc

    return {"grade": result}


# ---------------------------------------------------------------------------
# Entry point — lets you run `python server.py` directly during development.
# For production / HF Spaces use:  uvicorn server:app --host 0.0.0.0 --port 7860
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=True)
