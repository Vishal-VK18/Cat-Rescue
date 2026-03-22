# 🐱 Cat Rescue — OpenEnv Environment

> **Meta × PyTorch OpenEnv Hackathon by Scaler**
> A grid-based reinforcement learning environment where an AI agent navigates rooms to rescue trapped cats.

---

## 🎮 Game Concept

The agent starts at the top-left corner of a grid and must reach every cat before the step budget runs out. Static walls block movement on higher levels, and moving obstacles patrol the grid on the hardest level.

| Level | Grid | Cats | Obstacles |
|-------|------|------|-----------|
| 1 | 3 × 3 | 1 | None |
| 2 | 5 × 5 | 2 | Static walls |
| 3 | 7 × 7 | 3 | Static walls + 2 moving |

### Actions
| Integer | Direction |
|---------|-----------|
| `0` | UP |
| `1` | DOWN |
| `2` | LEFT |
| `3` | RIGHT |

### Rewards
| Event | Reward |
|-------|--------|
| Every step taken | `−0.1` |
| Hit a wall / boundary | `−0.3` |
| Rescue a cat | `+1.0` |
| All cats rescued (bonus) | `+2.0` |

---

## 📁 Project Structure

```
Cat-Rescue/
├── environment.py   # Core game logic — CatRescueEnv
├── rewards.py       # Reward schedule — CatRescueRewards
├── grader.py        # Episode scoring — CatRescueGrader
├── server.py        # FastAPI server (OpenEnv HTTP API)
├── Dockerfile       # Container build for HF Spaces
└── requirements.txt # Python dependencies
```

---

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860
```

Open the interactive API docs at **[http://localhost:7860/docs](http://localhost:7860/docs)**

---

## 🐳 Running with Docker

```bash
# Build the image
docker build -t cat-rescue .

# Run the container
docker run -p 7860:7860 cat-rescue
```

---

## 🌐 API Endpoints

### `GET /`
Health check — confirms the server is up.

### `POST /reset`
Start a new episode. Re-initialises the environment.

**Request body:**
```json
{ "level": 1, "max_steps": 200, "seed": 42 }
```

### `POST /step`
Take one action. Returns the new observation, reward, done flag, and info dict.

**Request body:**
```json
{ "action": 0 }
```

**Response:**
```json
{
  "observation": { "grid": [...], "agent_pos": [0,1], "cats_rescued": 0, ... },
  "reward": -0.1,
  "done": false,
  "info": { "event": "step", "cats_rescued": 0, "step_count": 1 }
}
```

### `GET /state`
Read the current grid state **without** advancing the episode.

### `POST /grade`
Score a completed episode log.

**Request body:**
```json
{
  "episode_log": [
    { "action": 1, "reward": -0.1, "done": false, "info": { ... } }
  ]
}
```

---

## 🤖 Quickstart Agent Loop

```python
import requests

BASE = "http://localhost:7860"

# 1. Start a new episode on level 2
obs = requests.post(f"{BASE}/reset", json={"level": 2, "seed": 0}).json()

done = False
episode_log = []

# 2. Run a random agent
while not done:
    action = random.randint(0, 3)           # replace with your policy
    resp = requests.post(f"{BASE}/step", json={"action": action}).json()
    done = resp["done"]
    episode_log.append({
        "action": action,
        "reward": resp["reward"],
        "done":   resp["done"],
        "info":   resp["info"],
    })

# 3. Grade the episode
result = requests.post(f"{BASE}/grade", json={"episode_log": episode_log}).json()
print(result)
```

---

## 🏗️ Deploying to Hugging Face Spaces

1. Create a new Space with **Docker** as the SDK.
2. Push this repository to the Space.
3. HF Spaces automatically builds the Dockerfile and routes traffic to port `7860`.

---

## 📜 License

MIT — built for the **Meta × PyTorch OpenEnv Hackathon by Scaler**.
