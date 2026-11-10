"""Deep Q-Network (DQN) for CartPole-v1 and MsPacman-v5.

Pieces (all in this file):
  - ReplayBuffer       stores (state, action, reward, next state, terminal) and hands out random batches
  - MLP / AtariNet     the neural network that estimates Q(state, action) for every action
  - train()            plays episodes with epsilon-greedy actions, learns from the replay buffer,
                       and copies the network into a slower-moving "target network" now and then
  - evaluate()         plays episodes with the trained network and returns the scores

Run:  python dqn.py --game cartpole --out runs/cartpole_seed0
      python dqn.py --game pacman   --out runs/pacman_seed0 --device cuda
"""
import argparse
import copy
import json
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)

MSPACMAN_COLOR = 210 + 164 + 74


def preprocess_observation(obs):
    """The function given in the assignment: 210x160x3 colour frame -> 88x80x1 numbers from -128 to 127."""
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == MSPACMAN_COLOR] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


# --------------------------------------------------------------------------------------- settings per game
CONFIGS = {
    "cartpole": dict(env="CartPole-v1", gamma=0.95, lr=5e-4, batch=64, buffer=50_000, learning_starts=1_000,
                     train_freq=1, target_every=250, tau=0.0, eps_start=1.0, eps_end=0.02, eps_decay_steps=10_000,
                     episodes=800, hidden=128, double=True, clip_rewards=False, eval_eps=0.0, grad_clip=10.0,
                     eval_every=10, eval_episodes=30),
    "pacman": dict(env="ALE/MsPacman-v5", gamma=0.99, lr=1e-4, batch=32, buffer=100_000, learning_starts=10_000,
                   train_freq=4, target_every=2_000, tau=0.0, eps_start=1.0, eps_end=0.05, eps_decay_steps=400_000,
                   episodes=6000, hidden=512, double=True, clip_rewards=True, eval_eps=0.05, grad_clip=10.0,
                   eval_every=0, eval_episodes=0),
}


# --------------------------------------------------------------------------------------- states
class FrameStack:
    """Pacman: the network sees the last 4 cleaned-up frames at once (so it can tell which way things move).
    At the start of a game the first frame is simply repeated."""

    def __init__(self, n=4):
        self.n, self.frames = n, None

    def reset(self, obs):
        frame = preprocess_observation(obs)[:, :, 0]
        self.frames = np.repeat(frame[None], self.n, axis=0)
        return self.frames.copy()

    def step(self, obs):
        frame = preprocess_observation(obs)[:, :, 0]
        self.frames = np.concatenate([self.frames[1:], frame[None]], axis=0)
        return self.frames.copy()


class Game:
    """One game with the same small interface for CartPole and Pacman: reset() / step(action)."""

    def __init__(self, game, seed=0, max_steps=None):
        self.game, self.seed, self.first = game, seed, True
        self.env = gym.make(CONFIGS[game]["env"], **({"max_episode_steps": max_steps} if max_steps else {}))  # max_steps: for tests only
        self.stack = FrameStack(4) if game == "pacman" else None
        self.n_actions = self.env.action_space.n
        self.env.action_space.seed(seed)

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed if self.first else None)
        self.first = False
        return self.stack.reset(obs) if self.stack else obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(int(action))
        state = self.stack.step(obs) if self.stack else obs.astype(np.float32)
        return state, float(reward), bool(terminated), bool(truncated)


# --------------------------------------------------------------------------------------- replay buffer
class ReplayBuffer:
    """A fixed-size memory of past steps. When it is full the oldest step is overwritten.
    `terminal` is 1 only when the game really ended (the pole fell, Pacman lost her last life); a game stopped by
    the time limit is NOT terminal, because the future after the limit would still have had reward."""

    def __init__(self, capacity, state_shape, state_dtype):
        self.capacity = capacity
        self.s = np.zeros((capacity, *state_shape), state_dtype)
        self.s2 = np.zeros((capacity, *state_shape), state_dtype)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.terminal = np.zeros(capacity, np.float32)
        self.born = np.zeros(capacity, np.int64)  # number of the step at which this tuple was stored (to know its age)
        self.pos, self.size, self.total = 0, 0, 0

    def add(self, s, a, r, s2, terminal):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.terminal[i], self.born[i] = s, a, r, s2, float(terminal), self.total
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total += 1

    def sample(self, batch, rng):
        idx = rng.integers(0, self.size, batch)  # only slots that have really been filled
        return idx, self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.terminal[idx]


# --------------------------------------------------------------------------------------- networks
class MLP(nn.Module):
    """CartPole: 4 state numbers -> two hidden layers -> one Q value per action."""

    def __init__(self, n_in, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class AtariNet(nn.Module):
    """Pacman: 4 stacked 88x80 pictures -> three convolution layers -> one Q value per action (9 actions)."""

    def __init__(self, n_actions, hidden=512):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 4, 88, 80)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.head(self.conv(x.float() / 128.0))  # pictures run from -128 to 127: scale to about -1..1


def make_network(game, n_actions, hidden):
    return MLP(4, n_actions, hidden) if game == "cartpole" else AtariNet(n_actions, hidden)


# --------------------------------------------------------------------------------------- learning step
def epsilon_at(step, start, end, decay_steps):
    """Chance of a random action: falls in a straight line from `start` to `end` over `decay_steps` steps."""
    return max(end, start + (end - start) * min(step / decay_steps, 1.0))


def td_targets(online, target, r, s2, terminal, gamma, double=True):
    """The number each Q(s, a) is pulled towards:  reward + gamma * (value of the next state), or just the reward
    if the game ended. The next state is valued by the slow-moving target network. With `double`, the online
    network picks the best next action and the target network values it (this reduces over-estimation)."""
    with torch.no_grad():
        if double:
            best = online(s2).argmax(dim=1, keepdim=True)
            next_value = target(s2).gather(1, best).squeeze(1)
        else:
            next_value = target(s2).max(dim=1).values
        return r + gamma * (1.0 - terminal) * next_value


def learn(online, target, optimizer, batch, gamma, double, grad_clip, device):
    s, a, r, s2, terminal = [torch.as_tensor(x, device=device) for x in batch]
    y = td_targets(online, target, r, s2, terminal, gamma, double)
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y)  # Huber loss: squared for small errors, straight line for big ones
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def sync_target(online, target, tau=0.0):
    """Copy the online network into the target network (tau = 0: all at once; tau > 0: move a small fraction)."""
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.copy_(op if tau == 0 else (1 - tau) * tp + tau * op)


# --------------------------------------------------------------------------------------- playing
def greedy_action(net, state, device):
    with torch.no_grad():
        q = net(torch.as_tensor(state, device=device).unsqueeze(0))[0]
    return int(q.argmax()), float(q.max())


def discounted_returns(rewards, gamma, bootstrap=0.0):
    """Reward-to-go for every step of a finished game (bootstrap = value to add after the last step)."""
    out, running = np.zeros(len(rewards)), bootstrap
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def evaluate(net, game, episodes=500, seed=12345, eps=0.0, device="cpu", record_q=False):
    """Play `episodes` games with the trained network (greedy, or epsilon-greedy with a small eps).
    Returns the score of every game (and, if record_q, the predicted best Q and the realised return of every step)."""
    cfg = CONFIGS[game]
    env, rng = Game(game, seed=seed), np.random.default_rng(seed)
    net.eval()
    scores, predicted, realised = [], [], []
    for _ in range(episodes):
        state, done, total, qs, rs, truncated = env.reset(), False, 0.0, [], [], False
        while not done:
            action, qmax = greedy_action(net, state, device)
            if eps > 0 and rng.random() < eps:
                action = int(rng.integers(env.n_actions))
            state, reward, terminated, truncated = env.step(action)
            total += reward
            qs.append(qmax)
            rs.append(max(-1.0, min(1.0, reward)) if cfg["clip_rewards"] else reward)
            done = terminated or truncated
        scores.append(total)
        if record_q:
            bootstrap = (1.0 / (1 - cfg["gamma"])) if (truncated and game == "cartpole") else 0.0
            predicted += qs
            realised += list(discounted_returns(rs, cfg["gamma"], bootstrap))
    net.train()
    scores = np.array(scores)
    return (scores, np.array(predicted), np.array(realised)) if record_q else scores


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - window)]) / (i + 1 - max(0, i + 1 - window)) for i in range(len(x))])


# --------------------------------------------------------------------------------------- training
def train(game, seed=0, out=None, device="cpu", time_budget=None, snapshots=(), **overrides):
    cfg = {**CONFIGS[game], **overrides}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = Game(game, seed=seed, max_steps=cfg.get("max_steps"))
    online = make_network(game, env.n_actions, cfg["hidden"]).to(device)
    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)  # the target network is never trained directly
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg["lr"])
    state_shape = (4,) if game == "cartpole" else (4, 88, 80)
    state_dtype = np.float32 if game == "cartpole" else np.int8
    buffer = ReplayBuffer(cfg["buffer"], state_shape, state_dtype)
    # optional second memory that only keeps steps where the game really ended; a fixed share of every batch comes from it,
    # so the network does not forget how bad failing is once failures become rare
    fail_share = cfg.get("failure_fraction", 0.0)
    failures = ReplayBuffer(5_000, state_shape, state_dtype) if fail_share > 0 else None
    n_fail = int(round(cfg["batch"] * fail_share))
    lr_end, lr_decay_steps = cfg.get("lr_end"), cfg.get("lr_decay_steps", 100_000)

    log = dict(reward=[], length=[], max_q=[], mean_q=[], epsilon=[], loss=[], step=[])
    best_avg, best_val, step, t0, losses = -np.inf, -np.inf, 0, time.time(), []
    val_log = dict(episode=[], score=[])
    if out:
        os.makedirs(out, exist_ok=True)
    for episode in range(1, cfg["episodes"] + 1):
        state, done, total, length, qs = env.reset(), False, 0.0, 0, []
        while not done:
            eps = epsilon_at(step, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
            action, qmax = greedy_action(online, state, device)
            qs.append(qmax)
            if rng.random() < eps:
                action = int(rng.integers(env.n_actions))
            next_state, reward, terminated, truncated = env.step(action)
            stored = max(-1.0, min(1.0, reward)) if cfg["clip_rewards"] else reward
            buffer.add(state, action, stored, next_state, terminated)  # a time-limit stop is NOT terminal
            if failures is not None and terminated:
                failures.add(state, action, stored, next_state, terminated)
            state, done = next_state, terminated or truncated
            total += reward
            length += 1
            step += 1
            if buffer.size >= cfg["learning_starts"] and step % cfg["train_freq"] == 0:
                if failures is not None and failures.size >= n_fail:
                    main, extra = buffer.sample(cfg["batch"] - n_fail, rng)[1:], failures.sample(n_fail, rng)[1:]
                    batch = tuple(np.concatenate([m, e]) for m, e in zip(main, extra))
                else:
                    batch = buffer.sample(cfg["batch"], rng)[1:]
                if lr_end is not None:  # learning rate scheduler: straight line from lr down to lr_end
                    optimizer.param_groups[0]["lr"] = cfg["lr"] + (lr_end - cfg["lr"]) * min(step / lr_decay_steps, 1.0)
                losses.append(learn(online, target, optimizer, batch, cfg["gamma"], cfg["double"], cfg["grad_clip"], device))
            if cfg["tau"] > 0:
                sync_target(online, target, cfg["tau"])
            elif cfg["target_every"] and step % cfg["target_every"] == 0:
                sync_target(online, target)
        log["reward"].append(total); log["length"].append(length); log["max_q"].append(max(qs)); log["mean_q"].append(float(np.mean(qs)))
        log["epsilon"].append(eps); log["loss"].append(float(np.mean(losses)) if losses else float("nan")); log["step"].append(step)
        losses = []
        avg100 = float(np.mean(log["reward"][-100:]))
        if out:
            if episode in snapshots:
                torch.save(online.state_dict(), os.path.join(out, f"model_ep{episode}.pt"))
            if cfg["eval_every"]:
                # choose the model by playing a few greedy games with fixed starting states (separate from the final test games)
                if episode % cfg["eval_every"] == 0:
                    val = float(evaluate(online, game, cfg["eval_episodes"], seed=424242, eps=cfg["eval_eps"], device=device).mean())
                    val_log["episode"].append(episode); val_log["score"].append(val)
                    if val > best_val:
                        best_val = val
                        torch.save(online.state_dict(), os.path.join(out, "model_best.pt"))
                if avg100 > best_avg:
                    best_avg = avg100
            elif episode >= min(100, cfg["episodes"]) and avg100 > best_avg:  # keep the best model by training average
                best_avg = avg100
                torch.save(online.state_dict(), os.path.join(out, "model_best.pt"))
            if episode % 25 == 0 or episode == cfg["episodes"]:
                np.savez(os.path.join(out, "log.npz"), **{k: np.array(v) for k, v in log.items()}, val_episode=np.array(val_log["episode"]), val_score=np.array(val_log["score"]))
                torch.save(online.state_dict(), os.path.join(out, "model_last.pt"))
        if episode % 25 == 0:
            print(f"episode {episode:5d} | step {step:8d} | eps {eps:.3f} | reward {total:7.1f} | avg100 {avg100:7.1f} | "
                  f"max Q {log['max_q'][-1]:6.2f} | loss {log['loss'][-1]:.4f} | {time.time() - t0:7.0f}s", flush=True)
        if time_budget and time.time() - t0 > time_budget:
            print("time budget reached, stopping", flush=True)
            break
    if out:
        np.savez(os.path.join(out, "log.npz"), **{k: np.array(v) for k, v in log.items()}, val_episode=np.array(val_log["episode"]), val_score=np.array(val_log["score"]))
        torch.save(online.state_dict(), os.path.join(out, "model_last.pt"))
        json.dump(dict(cfg=cfg, seed=seed, episodes_played=len(log["reward"]), steps=step, seconds=time.time() - t0,
                       best_avg100=best_avg, best_validation=best_val), open(os.path.join(out, "run_info.json"), "w"), indent=1)
    return (online, log, buffer) if cfg.get("return_buffer") else (online, log)  # return_buffer: for tests only


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=list(CONFIGS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--time-budget", type=float, help="stop after this many seconds")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--snapshots", default="", help="comma-separated episode numbers at which to save the model")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    extra = {"episodes": a.episodes} if a.episodes else {}
    snaps = tuple(int(x) for x in a.snapshots.split(",") if x)
    train(a.game, a.seed, a.out, a.device, a.time_budget, snaps, **extra)
