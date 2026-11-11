"""CartPole tuning: tries a few settings on several seeds each and reports how well the chosen network plays.
Run: python experiments/sweep_cartpole.py        (about 10 minutes; results in sweep_cartpole.csv)
The score of a setting is judged on 100 greedy test games with their own random seed (777), not on the training games.

History (the earlier rounds are kept as sweep_cartpole_firstrun.csv and sweep_cartpole_secondrun.csv):
  round 1: settings A-F (soft or hard target update, learning rate, buffer size, 1000 games), model chosen by the training average.
  round 2: settings G-K (a share of game-ending steps in every batch, a falling learning rate, a bigger network).
           Result of both rounds: most runs level off at 100-250 and a few reach about 500; the weak ones fail because the
           cart leaves the track (diagnose_cartpole.py), not because the pole falls.
  round 3 (this file): the model is now chosen by playing 30 greedy games every 10 games of training (a fixed set of
           starting states, not the test games), and settings that did best so far are compared on 5 other seeds."""
import csv
import os
from multiprocessing import Pool

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import dqn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "experiments")
SETTINGS = {
    "B. hard target update every 500 steps (default)": dict(),
    "H. as B plus 25% of every batch from a memory of game-ending steps": dict(failure_fraction=0.25),
    "I. as B plus learning rate falling from 1e-3 to 1e-5": dict(lr=1e-3, lr_end=1e-5, lr_decay_steps=100_000),
    "M. as B with the target update every 250 steps": dict(target_every=250),
    "N. soft target update (0.01), as in the first try": dict(tau=0.01, target_every=0),
}
SEEDS = (100, 101, 102, 103, 104)      # seeds used only for choosing the setting; the final runs use other seeds


def one(job):
    name, seed = job
    torch.set_num_threads(1)
    out = os.path.join(HERE, "runs_sweep", f"{name[0]}_seed{seed}")
    net, log = dqn.train("cartpole", seed=seed, out=out, **SETTINGS[name])
    hidden = SETTINGS[name].get("hidden", dqn.CONFIGS["cartpole"]["hidden"])
    final = dqn.evaluate(net, "cartpole", episodes=100, seed=777).mean()
    best_net = dqn.make_network("cartpole", 2, hidden)
    best_net.load_state_dict(torch.load(os.path.join(out, "model_best.pt")))
    best = dqn.evaluate(best_net, "cartpole", episodes=100, seed=777).mean()
    return name, seed, float(np.mean(log["reward"][-100:])), float(final), float(best)


if __name__ == "__main__":
    jobs = [(n, s) for n in SETTINGS for s in SEEDS]
    with Pool(min(12, os.cpu_count() or 4)) as pool:
        rows = pool.map(one, jobs)
    with open(os.path.join(OUT, "sweep_cartpole.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "seed", "last_100_training_games", "final_network_100_test_games", "chosen_network_100_test_games"])
        w.writerows(rows)
    for name in SETTINGS:
        r = [x for x in rows if x[0] == name]
        print(f"{name:70s} final network {[round(x[3]) for x in r]}  chosen network {[round(x[4]) for x in r]}  mean chosen {np.mean([x[4] for x in r]):.0f}")
