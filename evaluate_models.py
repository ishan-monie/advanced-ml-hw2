"""Plays 500 test games with a trained network and saves the scores (and, for every step, the predicted best Q
and the return that was actually earned afterwards).
Run: python evaluate_models.py --game cartpole --model runs/cartpole_seed0/model_best.pt --out results/cartpole_eval.npz
     python evaluate_models.py --game pacman   --model runs/pacman_seed0/model_best.pt   --out results/pacman_eval.npz --device cuda
The test games use their own random seed (12345), different from the training seed."""
import argparse
import os

import numpy as np
import torch

import dqn

ap = argparse.ArgumentParser()
ap.add_argument("--game", choices=list(dqn.CONFIGS), required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=500)
ap.add_argument("--eps", type=float, help="chance of a random action in the test games (default: 0 for CartPole, 0.05 for Pacman)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()

torch.set_num_threads(a.threads)
cfg = dqn.CONFIGS[a.game]
eps = cfg["eval_eps"] if a.eps is None else a.eps
net = dqn.make_network(a.game, 2 if a.game == "cartpole" else 9, cfg["hidden"]).to(a.device)
net.load_state_dict(torch.load(a.model, map_location=a.device))
scores, predicted, realised = dqn.evaluate(net, a.game, episodes=a.episodes, eps=eps, device=a.device, record_q=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, scores=scores, predicted=predicted, realised=realised, eps=eps)
print(f"{a.game}: {a.episodes} test games (random-action chance {eps}): mean {scores.mean():.2f}, std {scores.std():.2f}, "
      f"min {scores.min():.0f}, max {scores.max():.0f}")
