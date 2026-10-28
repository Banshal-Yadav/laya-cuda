"""FastAPI + WebSocket server for Cyberpunk 1v1 Pong with Laya Tactical AI."""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys
import time
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
# Pong Physics Engine
# ---------------------------------------------------------------------------

COURT_W = 850
COURT_H = 520
PADDLE_W = 14
PADDLE_H = 85
BALL_RADIUS = 8
BASE_BALL_SPEED = 9.0
MAX_BALL_SPEED = 18.0

class PongPhysics:
    def __init__(self):
        self.reset_match()

    def reset_match(self):
        self.p1_score = 0
        self.ai_score = 0
        self.p1_y = COURT_H / 2 - PADDLE_H / 2
        self.ai_y = COURT_H / 2 - PADDLE_H / 2
        self.ai_target_y = COURT_H / 2 - PADDLE_H / 2
        self.rally = 0
        self.max_rally = 0
        self.last_scorer = None
        self.reset_ball(direction=1)

    def reset_ball(self, direction: int = 1):
        self.bx = COURT_W / 2
        self.by = COURT_H / 2
        angle = random.uniform(-0.4, 0.4)
        speed = BASE_BALL_SPEED
        self.bvx = direction * speed * math.cos(angle)
        self.bvy = speed * math.sin(angle)
        self.speed = speed
        self.rally = 0

    def update(self, p1_target_y: float, ai_tactic: str, dt: float = 1.0 / 60.0) -> Optional[str]:
        """Steps physics. Returns 'p1' if player scored, 'ai' if AI scored, else None."""
        # 1. Update Player Paddle (smooth clamp)
        target = max(10, min(COURT_H - PADDLE_H - 10, p1_target_y - PADDLE_H / 2))
        self.p1_y += (target - self.p1_y) * 0.35

        # 2. Update AI Paddle (Laya tactical tracking)
        # Calculate where ball will arrive
        if self.bvx > 0:
            # Ball approaching AI: intercept target
            # Ball projection with wall bounces
            sim_x, sim_y = self.bx, self.by
            sim_vy = self.bvy
            while sim_x < COURT_W - 40:
                sim_x += self.bvx * 0.5
                sim_y += sim_vy * 0.5
                if sim_y <= BALL_RADIUS or sim_y >= COURT_H - BALL_RADIUS:
                    sim_vy = -sim_vy

            # AI Tactical positioning based on Laya choice
            tactical_offset = 0
            if ai_tactic == "SMASH_HIGH":
                tactical_offset = -PADDLE_H * 0.35  # Hit with top edge to cut ball upwards
            elif ai_tactic == "SMASH_LOW":
                tactical_offset = PADDLE_H * 0.35   # Hit with bottom edge to cut ball downwards
            elif ai_tactic == "SPIN_CUT":
                tactical_offset = random.choice([-1, 1]) * PADDLE_H * 0.4
            
            desired_ai_y = sim_y - PADDLE_H / 2 + tactical_offset
        else:
            # Ball moving away: return towards court center
            desired_ai_y = COURT_H / 2 - PADDLE_H / 2

        desired_ai_y = max(10, min(COURT_H - PADDLE_H - 10, desired_ai_y))
        ai_speed = 7.5 + min(self.rally * 0.25, 4.0)
        dy = desired_ai_y - self.ai_y
        if abs(dy) > ai_speed:
            self.ai_y += math.copysign(ai_speed, dy)
        else:
            self.ai_y = desired_ai_y

        # 3. Update Ball Position
        self.bx += self.bvx
        self.by += self.bvy

        # Top/Bottom wall bounce
        if self.by <= BALL_RADIUS:
            self.by = BALL_RADIUS
            self.bvy = abs(self.bvy)
        elif self.by >= COURT_H - BALL_RADIUS:
            self.by = COURT_H - BALL_RADIUS
            self.bvy = -abs(self.bvy)

        # 4. Paddle Collisions
        # Left Paddle (Player)
        p1_x = 30
        if (
            self.bx - BALL_RADIUS <= p1_x + PADDLE_W
            and self.bx + BALL_RADIUS >= p1_x
            and self.p1_y - 8 <= self.by <= self.p1_y + PADDLE_H + 8
            and self.bvx < 0
        ):
            self.bx = p1_x + PADDLE_W + BALL_RADIUS
            offset = (self.by - (self.p1_y + PADDLE_H / 2)) / (PADDLE_H / 2)
            bounce_angle = offset * 0.95  # Max ~55 degrees
            self.speed = min(MAX_BALL_SPEED, self.speed + 0.35)
            self.bvx = abs(self.speed * math.cos(bounce_angle))
            self.bvy = self.speed * math.sin(bounce_angle)
            self.rally += 1
            self.max_rally = max(self.max_rally, self.rally)

        # Right Paddle (Laya AI)
        ai_x = COURT_W - 30 - PADDLE_W
        if (
            self.bx + BALL_RADIUS >= ai_x
            and self.bx - BALL_RADIUS <= ai_x + PADDLE_W
            and self.ai_y - 8 <= self.by <= self.ai_y + PADDLE_H + 8
            and self.bvx > 0
        ):
            self.bx = ai_x - BALL_RADIUS
            offset = (self.by - (self.ai_y + PADDLE_H / 2)) / (PADDLE_H / 2)
            bounce_angle = offset * 0.95
            self.speed = min(MAX_BALL_SPEED, self.speed + 0.45)
            self.bvx = -abs(self.speed * math.cos(bounce_angle))
            self.bvy = self.speed * math.sin(bounce_angle)
            self.rally += 1
            self.max_rally = max(self.max_rally, self.rally)

        # 5. Scoring Check
        if self.bx < -20:
            # AI Scores!
            self.ai_score += 1
            self.last_scorer = "ai"
            self.reset_ball(direction=-1)
            return "ai"
        elif self.bx > COURT_W + 20:
            # Player Scores!
            self.p1_score += 1
            self.last_scorer = "player"
            self.reset_ball(direction=1)
            return "player"

        return None

    def get_tactical_state_prompt(self) -> str:
        py_center = self.p1_y + PADDLE_H / 2
        ai_center = self.ai_y + PADDLE_H / 2
        player_zone = "HIGH (top court)" if py_center < COURT_H * 0.38 else ("LOW (bottom court)" if py_center > COURT_H * 0.62 else "CENTER")
        open_zone = "BOTTOM CORNER exposed" if py_center < COURT_H * 0.5 else "TOP CORNER exposed"
        
        return (
            f"Ball speed {self.speed:.1f} at ({int(self.bx)}, {int(self.by)}). "
            f"Player paddle is at {player_zone}. {open_zone}. "
            f"Rally count: {self.rally}. Score AI {self.ai_score} vs Player {self.p1_score}."
        )


# ---------------------------------------------------------------------------
# Laya Tactical AI Engine (RTX CUDA Graph)
# ---------------------------------------------------------------------------

TACTICS = ["SMASH_HIGH", "SMASH_LOW", "FLAT_CENTER", "SPIN_CUT"]

class LayaPongAI:
    def __init__(self, agent: laya.Agent):
        self.engine = FastDecisionEngine(agent, options=TACTICS, max_seq_len=64)
        self.options = TACTICS
        self.temp = float(agent._temperatures[self.engine.norm_q["qtype"]]) if hasattr(agent, "_temperatures") else 1.0
        # Warmup evaluate
        dummy = "Ball speed 9.0 at (400, 250). Player paddle is at CENTER. Rally count: 0. Score AI 0 vs Player 0."
        for _ in range(5):
            self.evaluate_tactic(dummy, COURT_H / 2)

    def evaluate_tactic(self, prompt: str, player_y: float) -> Tuple[str, Dict[str, float], float, str]:
        t0 = time.perf_counter()
        self.engine._load_state_into_buffers(prompt)
        self.engine.graph.replay()
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        scores = self.engine.static_scores[0, :len(self.options)].float()
        probs_tensor = torch.softmax(scores / max(self.temp, 0.1), dim=-1)
        probs = {opt: float(probs_tensor[i].item()) for i, opt in enumerate(self.options)}

        # Strategic weighting: exploit player's exposed side
        py_center = player_y + PADDLE_H / 2
        weighted_scores = {}
        for opt in self.options:
            val = probs[opt]
            if py_center < COURT_H * 0.45 and opt == "SMASH_LOW":
                val += 1.5  # Player is high, smash low
            elif py_center > COURT_H * 0.55 and opt == "SMASH_HIGH":
                val += 1.5  # Player is low, smash high
            elif opt == "SPIN_CUT":
                val += 0.5
            weighted_scores[opt] = val

        best_tactic = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])

        comments = {
            "SMASH_HIGH": "Exploiting open high corner with sharp upward angle",
            "SMASH_LOW": "Targeting exposed low baseline with downward slice",
            "FLAT_CENTER": "Driving high-velocity center shot to test player reaction",
            "SPIN_CUT": "Applying sharp angle cut off paddle edge",
        }
        comment = comments.get(best_tactic, "Tracking ball trajectory")

        return best_tactic, probs, latency_ms, comment


# ---------------------------------------------------------------------------
# FastAPI & WebSocket
# ---------------------------------------------------------------------------

app = FastAPI(title="Laya Cyberpunk Pong")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

pong = PongPhysics()
ai_brain: Optional[LayaPongAI] = None
p1_target_mouse_y = COURT_H / 2

@app.on_event("startup")
def startup_event():
    global ai_brain
    print("[Pong Server] Loading Laya model on RTX GPU...")
    agent = laya.load("convaiinnovations/laya", dtype="float16")
    ai_brain = LayaPongAI(agent)
    print("[Pong Server] Laya Pong Tactical Engine ready.")

@app.get("/")
def get_index():
    index_file = static_dir / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global p1_target_mouse_y, pong
    await websocket.accept()
    print("[WS] Player connected to Pong.")

    async def player_listener():
        global p1_target_mouse_y, pong
        try:
            while True:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                mtype = data.get("type")
                if mtype == "move":
                    p1_target_mouse_y = float(data.get("y", COURT_H / 2))
                elif mtype == "reset":
                    pong.reset_match()
        except Exception:
            pass

    listener_task = asyncio.create_task(player_listener())

    current_tactic = "FLAT_CENTER"
    tactic_probs = {t: 0.25 for t in TACTICS}
    latency_ms = 7.0
    comment = "Ready to serve"

    try:
        frame_idx = 0
        while True:
            t_frame_start = time.perf_counter()

            # Evaluate tactic every 10 frames or on direction reversal
            if frame_idx % 12 == 0 and ai_brain is not None and pong.bvx > 0:
                prompt = pong.get_tactical_state_prompt()
                current_tactic, tactic_probs, latency_ms, comment = ai_brain.evaluate_tactic(
                    prompt, pong.p1_y
                )

            # Step physics
            score_event = pong.update(p1_target_mouse_y, current_tactic)

            payload = {
                "type": "state",
                "court": {"w": COURT_W, "h": COURT_H},
                "ball": {"x": round(pong.bx, 1), "y": round(pong.by, 1), "speed": round(pong.speed, 1)},
                "p1": {"y": round(pong.p1_y, 1), "score": pong.p1_score},
                "ai": {"y": round(pong.ai_y, 1), "score": pong.ai_score, "tactic": current_tactic, "comment": comment},
                "probabilities": tactic_probs,
                "latency_ms": round(latency_ms, 1),
                "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
                "rally": pong.rally,
                "max_rally": pong.max_rally,
                "score_event": score_event,
                "prompt": pong.get_tactical_state_prompt(),
            }
            await websocket.send_json(payload)

            frame_idx += 1
            # 60 FPS tick
            elapsed = time.perf_counter() - t_frame_start
            sleep_time = max(0.001, (1.0 / 60.0) - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        print("[WS] Player disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        listener_task.cancel()


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8001, log_level="info")
