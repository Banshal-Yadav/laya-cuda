"""FastAPI + WebSocket live streaming server for Laya-powered Snake AI."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure root workspace is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import laya_cuda as laya
from laya_cuda.fast_engine import FastDecisionEngine

# ---------------------------------------------------------------------------
# Game Environment
# ---------------------------------------------------------------------------

class SnakeGame:
    DIRECTIONS = {
        "UP":    (0, -1),
        "DOWN":  (0, 1),
        "LEFT":  (-1, 0),
        "RIGHT": (1, 0),
    }
    OPPOSITES = {
        "UP": "DOWN", "DOWN": "UP",
        "LEFT": "RIGHT", "RIGHT": "LEFT",
    }

    def __init__(self, width: int = 16, height: int = 16):
        self.width = width
        self.height = height
        self.recent_positions = deque(maxlen=12)
        self.reset()

    def reset(self):
        cx, cy = self.width // 2, self.height // 2
        self.snake: List[Tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = "RIGHT"
        self.score = 0
        self.steps = 0
        self.alive = True
        self.recent_positions.clear()
        self.food = self._spawn_food()

    def _spawn_food(self) -> Tuple[int, int]:
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.snake
        ]
        return random.choice(empty) if empty else (0, 0)

    def get_safe_actions(self) -> List[str]:
        hx, hy = self.snake[0]
        safe = []
        for name, (dx, dy) in self.DIRECTIONS.items():
            if name == self.OPPOSITES.get(self.direction):
                continue  # Cannot reverse 180° into neck
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.snake[:-1]:
                safe.append(name)
        return safe

    def step(self, action: str) -> bool:
        if not self.alive:
            return False

        if action != self.OPPOSITES.get(self.direction):
            self.direction = action

        dx, dy = self.DIRECTIONS[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # Wall collision
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            self.alive = False
            return False

        # Body collision
        if (nx, ny) in self.snake[:-1]:
            self.alive = False
            return False

        self.snake.insert(0, (nx, ny))
        self.recent_positions.append((nx, ny))
        self.steps += 1

        if (nx, ny) == self.food:
            self.score += 1
            self.food = self._spawn_food()
            self.recent_positions.clear()
        else:
            self.snake.pop()

        return self.alive

    def get_state_prompt(self) -> str:
        hx, hy = self.snake[0]
        fx, fy = self.food
        curr_dist = abs(hx - fx) + abs(hy - fy)

        rel_x = "RIGHT" if fx > hx else ("LEFT" if fx < hx else "SAME_X")
        rel_y = "DOWN" if fy > hy else ("UP" if fy < hy else "SAME_Y")

        move_evals = []
        for name, (dx, dy) in self.DIRECTIONS.items():
            nx, ny = hx + dx, hy + dy
            is_wall = nx < 0 or nx >= self.width or ny < 0 or ny >= self.height
            is_body = (nx, ny) in self.snake[:-1]
            is_opp = (name == self.OPPOSITES.get(self.direction))
            if is_wall or is_body or is_opp:
                move_evals.append(f"{name} BLOCKED")
            else:
                new_dist = abs(nx - fx) + abs(ny - fy)
                if new_dist < curr_dist:
                    move_evals.append(f"{name} CLOSER to food")
                else:
                    move_evals.append(f"{name} OPEN")

        return (
            f"Snake head at ({hx}, {hy}), facing {self.direction}. "
            f"Food at ({fx}, {fy}) is {rel_x} and {rel_y}. "
            f"Surroundings: {', '.join(move_evals)}."
        )


# ---------------------------------------------------------------------------
# Fast Decision Engine Wrapper with Probabilities & Anti-Trap Shield
# ---------------------------------------------------------------------------

class LiveSnakeAI:
    def __init__(self, agent: laya.Agent):
        self.engine = FastDecisionEngine(agent, options=("UP", "DOWN", "LEFT", "RIGHT"))
        self.options = self.engine.options
        self.temp = float(agent._temperatures[self.engine.norm_q["qtype"]]) if hasattr(agent, "_temperatures") else 1.0

    def evaluate(
        self,
        state_text: str,
        safe_actions: List[str],
        head_pos: Tuple[int, int],
        food_pos: Tuple[int, int],
        recent_positions: deque
    ) -> Tuple[str, Dict[str, float], float, bool]:
        t0 = time.perf_counter()
        self.engine._load_state_into_buffers(state_text)
        self.engine.graph.replay()
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract raw scores & probabilities
        scores = self.engine.static_scores[0, :len(self.options)].float()
        probs_tensor = torch.softmax(scores / max(self.temp, 0.1), dim=-1)
        probs = {opt: float(probs_tensor[i].item()) for i, opt in enumerate(self.options)}

        raw_top_action = self.options[int(scores.argmax().item())]

        if not safe_actions:
            return raw_top_action, probs, latency_ms, False

        hx, hy = head_pos
        fx, fy = food_pos
        curr_dist = abs(hx - fx) + abs(hy - fy)

        best_act = raw_top_action if raw_top_action in safe_actions else safe_actions[0]
        best_val = -1e9
        is_shielded = False

        DIRECTIONS = SnakeGame.DIRECTIONS
        for act in safe_actions:
            dx, dy = DIRECTIONS[act]
            nx, ny = hx + dx, hy + dy
            new_dist = abs(nx - fx) + abs(ny - fy)

            # Combined utility: Laya model probability + food navigation gradient - loop penalty
            val = probs[act] * 2.0

            # Progress towards food
            if new_dist < curr_dist:
                val += 4.0
            else:
                val -= 1.0

            # Anti-loop / Anti-oscillation penalty: avoid recently visited cells
            if (nx, ny) in recent_positions:
                val -= 5.0

            if val > best_val:
                best_val = val
                best_act = act

        if best_act != raw_top_action:
            is_shielded = True

        return best_act, probs, latency_ms, is_shielded


# ---------------------------------------------------------------------------
# App & WebSocket Server
# ---------------------------------------------------------------------------

app = FastAPI(title="Laya Snake Live AI")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared state
game = SnakeGame(width=16, height=16)
ai_agent: Optional[LiveSnakeAI] = None
target_fps = 30
is_paused = False

@app.on_event("startup")
def startup_event():
    global ai_agent
    print("[Server] Loading Laya ModernBERT model on RTX GPU...")
    agent = laya.load("convaiinnovations/laya", dtype="float16")
    ai_agent = LiveSnakeAI(agent)
    print("[Server] Laya CUDA Graph decision engine initialized and ready.")

@app.get("/")
def get_index():
    index_file = static_dir / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global target_fps, is_paused, game
    await websocket.accept()
    print("[WS] Client connected.")

    # Background listener for client messages (avoids blocking / timeout jitter)
    async def listen_client():
        global target_fps, is_paused, game
        try:
            while True:
                raw_msg = await websocket.receive_text()
                msg = json.loads(raw_msg)
                mtype = msg.get("type")
                if mtype == "set_speed":
                    target_fps = max(1, min(120, int(msg.get("fps", 30))))
                elif mtype == "pause":
                    is_paused = True
                elif mtype == "resume":
                    is_paused = False
                elif mtype == "reset":
                    game.reset()
        except Exception:
            pass

    listener_task = asyncio.create_task(listen_client())

    try:
        while True:
            if not is_paused and ai_agent is not None:
                state_text = game.get_state_prompt()
                safe_acts = game.get_safe_actions()
                action, probs, latency_ms, is_shielded = ai_agent.evaluate(
                    state_text,
                    safe_acts,
                    game.snake[0],
                    game.food,
                    game.recent_positions
                )

                alive = game.step(action)
                if not alive:
                    # Auto-reset after a short pause
                    await websocket.send_json({
                        "type": "game_over",
                        "score": game.score,
                        "steps": game.steps,
                    })
                    await asyncio.sleep(0.8)
                    game.reset()
                    continue

                payload = {
                    "type": "tick",
                    "snake": game.snake,
                    "food": game.food,
                    "score": game.score,
                    "steps": game.steps,
                    "width": game.width,
                    "height": game.height,
                    "action": action,
                    "probabilities": probs,
                    "latency_ms": round(latency_ms, 2),
                    "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
                    "target_fps": target_fps,
                    "prompt": state_text,
                    "is_shielded": is_shielded,
                }
                await websocket.send_json(payload)

            # Frame rate delay
            sleep_sec = 1.0 / max(1, target_fps)
            await asyncio.sleep(sleep_sec)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        listener_task.cancel()


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
