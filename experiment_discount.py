"""Change-the-problem experiment for CartPole: does a longer-sighted discount fix the drifting-off-the-track failure?
Prediction (made before running): with discount 0.99 instead of 0.95 the network at the END of training plays much longer
games, because a failure that is 100 steps away is no longer invisible (0.95 ** 100 = 0.006, 0.99 ** 100 = 0.37).
Run: python experiment_discount.py       (about 5 minutes; results in experiment_discount.csv)
Both settings use the same 5 seeds (100-104) and everything else is identical (hard target update every 250 steps)."""
import csv
import os
from multiprocessing import Pool

import numpy as np
import torch

import dqn
import diagnose_cartpole

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (100, 101, 102, 103, 104)


def one(job):
    gamma, seed = job
    torch.set_num_threads(1)
    net, log = dqn.train("cartpole", seed=seed, gamma=gamma)          # no output folder: nothing saved
    score = float(dqn.evaluate(net, "cartpole", episodes=100, seed=777).mean())
    c, _ = diagnose_cartpole.causes(net)
    return gamma, seed, score, c["cart left the track"], c["pole fell"], c["reached 500 steps"]


if __name__ == "__main__":
    jobs = [(g, s) for g in (0.95, 0.99) for s in SEEDS]
    with Pool(10) as pool:
        rows = pool.map(one, jobs)
    with open(os.path.join(HERE, "experiment_discount.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["discount", "seed", "final_network_100_test_games", "cart_left_track", "pole_fell", "reached_500"])
        w.writerows(rows)
    for g in (0.95, 0.99):
        r = [x for x in rows if x[0] == g]
        print(f"discount {g}: final network scores {[round(x[2]) for x in r]}, mean {np.mean([x[2] for x in r]):.0f}; "
              f"games lost by leaving the track {sum(x[3] for x in r)}, by the pole falling {sum(x[4] for x in r)}, reached 500 {sum(x[5] for x in r)}")
