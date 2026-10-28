import time
import torch
import numpy as np
import laya_cuda as laya

agent = laya.load("convaiinnovations/laya", dtype="float16")

def make_snake_state(head, food, grid_size=10, body=None):
    if body is None:
        body = []
    hx, hy = head
    fx, fy = food
    
    # Check 1-step collisions
    danger = []
    directions = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    for name, (dx, dy) in directions.items():
        nx, ny = hx + dx, hy + dy
        if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size or (nx, ny) in body:
            danger.append(f"{name} is DANGEROUS/BLOCKED")
        else:
            danger.append(f"{name} is SAFE")
            
    danger_str = ", ".join(danger)
    rel_x = "RIGHT" if fx > hx else ("LEFT" if fx < hx else "SAME_X")
    rel_y = "DOWN" if fy > hy else ("UP" if fy < hy else "SAME_Y")
    
    return f"Snake head at ({hx}, {hy}). Food at ({fx}, {fy}). Food is {rel_x} and {rel_y}. Wall status: {danger_str}."

candidates = ["UP", "DOWN", "LEFT", "RIGHT"]
batch_items = []
for c in candidates:
    candidate_state = (
        f"Snake situation: head is at (5, 5). Target food is to the RIGHT at (8, 5). "
        f"Obstacle check: UP is blocked by wall. DOWN, LEFT, RIGHT are open. "
        f"Candidate move: {c}."
    )
    batch_items.append((
        candidate_state,
        {"good_move": {"type": "noul", "instructions": "Is this a valid, safe move that moves toward the target food?"}}
    ))

results = agent.batch_predict(batch_items)
for c, r in zip(candidates, results):
    p_true = r["details"]["good_move"]["p_true"]
    print(f"Candidate {c:5s} -> p_true = {p_true:.3f} ({'YES' if r['answers']['good_move'] else 'NO'})")

