"""A reproducible, real Q-learning smoke job; never described as VLA training."""
import argparse
import json
import random
import time

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=42)
p.add_argument("--episodes", type=int, default=200)
p.add_argument("--learning-rate", type=float, default=0.2)
a = p.parse_args()
rng = random.Random(a.seed)
q = [[0.0, 0.0] for _ in range(9)]
returns = []
print("Training tabular Q-learning on a 1D reach task. No robot, camera, or VLA is used.", flush=True)
for episode in range(a.episodes):
    state, total = rng.randrange(0, 8), 0
    epsilon = max(0.05, 0.9 * (1 - episode / max(1, a.episodes)))
    for step in range(25):
        action = rng.randrange(2) if rng.random() < epsilon else max(range(2), key=lambda x: q[state][x])
        nxt = max(0, min(8, state + (-1 if action == 0 else 1)))
        reward = 1.0 if nxt == 8 else -0.04
        target = reward + (0 if nxt == 8 else 0.95 * max(q[nxt]))
        q[state][action] += a.learning_rate * (target - q[state][action])
        state, total = nxt, total + reward
        if state == 8:
            break
    returns.append(total)
    if (episode + 1) % 5 == 0 or episode == a.episodes - 1:
        successes = 0
        for start in range(8):
            s = start
            for _ in range(25):
                act = max(range(2), key=lambda x: q[s][x])
                s = max(0, min(8, s + (-1 if act == 0 else 1)))
                if s == 8:
                    successes += 1
                    break
        print("ARMATURE_METRIC " + json.dumps({"step": episode + 1, "reward": sum(returns[-20:]) / len(returns[-20:]), "success_rate": successes / 8}), flush=True)
    time.sleep(0.025)
print("Finished. Success rate is greedy-policy evaluation over all 8 starting states.", flush=True)
