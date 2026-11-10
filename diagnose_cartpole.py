"""Why do some CartPole networks fail? Plays 100 greedy games with every saved network of the tuning runs and counts
how each game ended: the cart left the track, the pole fell, or the 500-step limit was reached.
Run: python diagnose_cartpole.py        (results in diagnose_cartpole.csv)"""
import csv
import glob
import os

import gymnasium as gym
import numpy as np
import torch

import dqn

HERE = os.path.dirname(os.path.abspath(__file__))


def causes(net, games=100):
    env = gym.make("CartPole-v1")
    out = {"cart left the track": 0, "pole fell": 0, "reached 500 steps": 0}
    lengths = []
    for g in range(games):
        state, _ = env.reset(seed=9000 + g)
        steps = 0
        while True:
            action, _ = dqn.greedy_action(net, state.astype(np.float32), "cpu")
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        lengths.append(steps)
        if steps >= 500:
            out["reached 500 steps"] += 1
        elif abs(state[0]) > 2.4:
            out["cart left the track"] += 1
        else:
            out["pole fell"] += 1
    return out, float(np.mean(lengths))


if __name__ == "__main__":
    torch.set_num_threads(1)
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "runs_sweep", "*", "model_last.pt"))):
        net = dqn.make_network("cartpole", 2, dqn.CONFIGS["cartpole"]["hidden"])
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        c, mean_len = causes(net)
        rows.append([os.path.basename(os.path.dirname(path)), round(mean_len, 1), c["cart left the track"], c["pole fell"], c["reached 500 steps"]])
        print(rows[-1])
    with open(os.path.join(HERE, "diagnose_cartpole.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "mean_game_length", "cart_left_track", "pole_fell", "reached_500"])
        w.writerows(rows)
    tot = np.array([r[2:] for r in rows]).sum(axis=0)
    print(f"over {len(rows)} networks x 100 games: cart left the track {tot[0]}, pole fell {tot[1]}, reached 500 steps {tot[2]}")
