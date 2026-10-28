import time
import torch
import laya_cuda as laya

agent = laya.load("convaiinnovations/laya", dtype="float16")
state = "Head: (5,5), Food: (5,8), Danger: UP, LEFT"
questions = {
    "next_move": {
        "type": "choice",
        "instructions": "Pick safe move towards food",
        "criteria": ["UP", "DOWN", "LEFT", "RIGHT"],
    }
}

# Warmup
for _ in range(5):
    agent.predict(state, questions)

# Run one predict to get a valid batch
norm_qs = laya.common.prepare_questions(questions)
state_ids = agent.tok(laya.common.serialize_state(state), add_special_tokens=False)
head_ids, markers = laya.common.build_prefix(agent.tok, norm_qs[0], max_len=512, head_max_len=192)
seq = laya.common.build_sequence(agent.tok, state_ids, head_ids, markers, max_len=512)
seq["qtype"] = norm_qs[0]["qtype"]
batch = laya.agent.collate_items([seq], pad_id=agent.tok.pad_token_id, pad_to_multiple=8)

t_ids  = torch.tensor(batch["input_ids"],      dtype=torch.long, device=agent.device)
t_mask = torch.tensor(batch["attention_mask"], dtype=torch.bool, device=agent.device)
t_mpos = torch.tensor(batch["marker_pos"],     dtype=torch.long, device=agent.device)
t_mmask= torch.tensor(batch["marker_mask"],    dtype=torch.bool, device=agent.device)
t_qt   = torch.tensor(batch["qtype"],          dtype=torch.long, device=agent.device)

# Warmup GPU
for _ in range(10):
    with torch.no_grad():
        agent.model(t_ids, t_mask, t_mpos, t_mmask, t_qt)
torch.cuda.synchronize()

# Benchmark raw GPU forward
times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        scores, act = agent.model(t_ids, t_mask, t_mpos, t_mmask, t_qt)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
raw_gpu_p50 = times[len(times)//2]
print(f"Sequence length: {t_ids.shape[1]} tokens")
print(f"Raw GPU model forward: {raw_gpu_p50:.2f} ms ({1000/raw_gpu_p50:.1f} FPS)")

# CUDA Graph capture!
# Static buffers
static_ids = t_ids.clone()
static_mask = t_mask.clone()
static_mpos = t_mpos.clone()
static_mmask = t_mmask.clone()
static_qt = t_qt.clone()

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        agent.model(static_ids, static_mask, static_mpos, static_mmask, static_qt)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    static_scores, static_act = agent.model(static_ids, static_mask, static_mpos, static_mmask, static_qt)

# Replay benchmark
g_times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    g_times.append((time.perf_counter() - t0) * 1000)

g_times.sort()
graph_p50 = g_times[len(g_times)//2]
print(f"CUDA Graph replay:     {graph_p50:.2f} ms ({1000/graph_p50:.1f} FPS)")


