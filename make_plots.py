"""Makes all the graphs of the report (saved in figures/).
Run: python make_plots.py
Needs: runs/<game>_seed*/log.npz (training logs), results/<game>_eval.npz (the 500 test games, from evaluate_models.py),
and for the CartPole policy map the saved networks runs/cartpole_seed0/model_ep*.pt and model_best.pt."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dqn

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
NAMES = {"cartpole": "CartPole-v1", "pacman": "MsPacman-v5"}
MAIN_SEED = 0


def load_log(game, seed):
    return dict(np.load(os.path.join(HERE, "runs", f"{game}_seed{seed}", "log.npz")))


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=150)
    plt.close()


def standard_plots(game):
    log = load_log(game, MAIN_SEED)
    n = len(log["reward"])
    episodes = np.arange(1, n + 1)
    gamma = dqn.CONFIGS[game]["gamma"]

    # (i) maximum Q value in each training episode
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, log["max_q"], color="C0", lw=0.9, label="largest Q value seen in the episode")
    plt.plot(episodes, dqn.moving_average(log["max_q"]), color="C3", lw=2, label="moving average of the last 100 episodes")
    if game == "cartpole":
        plt.axhline(1 / (1 - gamma), color="k", ls="--", lw=1, label=f"largest possible true value, 1/(1-{gamma}) = {1 / (1 - gamma):.0f}")
    plt.xlabel("training episode")
    plt.ylabel("maximum Q value" + (" (rewards clipped to -1..+1)" if game == "pacman" else ""))
    plt.title(f"{NAMES[game]}: maximum Q value during training")
    plt.grid(alpha=.3); plt.legend(fontsize=8)
    save(f"{game}_max_q.png")

    # (ii) episode rewards with the 100-episode moving average
    plt.figure(figsize=(7, 4))
    plt.plot(episodes, log["reward"], color="0.7", lw=0.7, label="reward of each episode")
    plt.plot(episodes, dqn.moving_average(log["reward"]), color="C3", lw=2, label="moving average of the last 100 episodes")
    plt.xlabel("training episode"); plt.ylabel("total reward in the episode" + (" (maximum 500)" if game == "cartpole" else ""))
    plt.title(f"{NAMES[game]}: reward during training")
    plt.grid(alpha=.3); plt.legend(fontsize=8)
    save(f"{game}_training.png")

    # (iii) histogram of the 500 test games
    ev = np.load(os.path.join(HERE, "results", f"{game}_eval.npz"))
    scores = ev["scores"]
    plt.figure(figsize=(6, 4))
    bins = np.arange(0, 511, 10) if game == "cartpole" else np.arange(0, scores.max() + 200, 200)
    plt.hist(scores, bins=bins, color="C0", edgecolor="w")
    plt.axvline(scores.mean(), color="C3", ls="--", label=f"mean {scores.mean():.1f}, std {scores.std():.1f}")
    plt.xlabel("total reward in the game"); plt.ylabel("number of games (out of 500)")
    plt.title(f"{NAMES[game]}: 500 test games with the trained network")
    plt.legend()
    save(f"{game}_histogram.png")

    # all seeds (how much does luck matter?)
    seeds = sorted(int(p.split("seed")[-1].split(os.sep)[0]) for p in glob.glob(os.path.join(HERE, "runs", f"{game}_seed*", "log.npz")))
    if len(seeds) > 1:
        plt.figure(figsize=(7, 4))
        for s in seeds:
            lg = load_log(game, s)
            plt.plot(np.arange(1, len(lg["reward"]) + 1), dqn.moving_average(lg["reward"]), lw=2 if s == MAIN_SEED else 1.2,
                     label=f"seed {s}" + (" (main)" if s == MAIN_SEED else ""))
        plt.xlabel("training episode"); plt.ylabel("moving average of the last 100 episodes")
        plt.title(f"{NAMES[game]}: different random seeds"); plt.grid(alpha=.3); plt.legend()
        save(f"{game}_seeds.png")


def policy_map():
    """Non-standard plot for CartPole: what the network 'thinks' the best push is, for every pole angle and pole speed,
    after different numbers of training games (cart in the middle and standing still)."""
    run = os.path.join(HERE, "runs", f"cartpole_seed{MAIN_SEED}")
    snaps = sorted(int(os.path.basename(p)[8:-3]) for p in glob.glob(os.path.join(run, "model_ep*.pt")))
    panels = [(f"after {e} games", os.path.join(run, f"model_ep{e}.pt")) for e in snaps] + [("final (best) network", os.path.join(run, "model_best.pt"))]
    angle = np.linspace(-0.21, 0.21, 121)
    speed = np.linspace(-2.5, 2.5, 121)
    A, W = np.meshgrid(angle, speed)
    states = np.zeros((A.size, 4), np.float32)
    states[:, 2], states[:, 3] = A.ravel(), W.ravel()
    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.0 * rows), squeeze=False)
    for ax, (title, path) in zip(axes.ravel(), panels):
        net = dqn.make_network("cartpole", 2, dqn.CONFIGS["cartpole"]["hidden"])
        net.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            q = net(torch.as_tensor(states)).numpy()
        diff = (q[:, 1] - q[:, 0]).reshape(A.shape)
        lim = np.abs(diff).max()
        im = ax.pcolormesh(angle, speed, diff, cmap="coolwarm", vmin=-lim, vmax=lim, shading="auto")
        ax.contour(angle, speed, diff, levels=[0], colors="k", linewidths=1)
        ax.axhline(0, color="0.5", lw=.5); ax.axvline(0, color="0.5", lw=.5)
        ax.set_title(title, fontsize=9); ax.set_xlabel("pole angle (radians)", fontsize=8); ax.set_ylabel("pole angular speed", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax).ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("CartPole, cart in the middle and standing still: Q(push right) - Q(push left).\n"
                 "Red: pushing right is better, blue: pushing left is better, black line: where the best push flips", fontsize=10)
    save("cartpole_policy_map.png")


def q_calibration(game):
    """Non-standard plot: the Q value the network predicted at every step of the test games against the discounted reward
    that was actually collected afterwards. On the diagonal = the network predicts correctly."""
    ev = np.load(os.path.join(HERE, "results", f"{game}_eval.npz"))
    pred, real = ev["predicted"], ev["realised"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pred), size=min(len(pred), 300_000), replace=False)
    plt.figure(figsize=(5.5, 5))
    hb = plt.hexbin(real[pick], pred[pick], gridsize=60, bins="log", cmap="viridis", mincnt=1)
    top = max(real.max(), pred.max())
    plt.plot([0, top], [0, top], color="C3", ls="--", label="perfect prediction")
    edges = np.linspace(real.min(), real.max() + 1e-9, 21)
    which = np.clip(np.digitize(real, edges[1:-1]), 0, 19)
    ks = [k for k in range(20) if (which == k).sum() >= 20]          # only bins with enough steps
    plt.plot([real[which == k].mean() for k in ks], [pred[which == k].mean() for k in ks], "o-", color="w", mec="k", ms=4, lw=1, label="average prediction")
    plt.xlabel("discounted reward actually collected afterwards"); plt.ylabel("maximum Q value predicted")
    plt.title(f"{NAMES[game]}: predicted against actual value\n(every step of the 500 test games)", fontsize=10)
    plt.colorbar(hb, label="number of steps (log scale)"); plt.legend(fontsize=8)
    save(f"{game}_q_calibration.png")


if __name__ == "__main__":
    for g in NAMES:
        if os.path.exists(os.path.join(HERE, "results", f"{g}_eval.npz")) and os.path.exists(os.path.join(HERE, "runs", f"{g}_seed{MAIN_SEED}", "log.npz")):
            standard_plots(g)
            if g == "pacman":     # in CartPole nearly every step has the same true value (20), so the plot says little
                q_calibration(g)
            print("plots made for", g)
    if glob.glob(os.path.join(HERE, "runs", "cartpole_seed0", "model_ep*.pt")):
        policy_map()
        print("policy map made")
