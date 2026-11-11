"""Checks for the DQN code: edge cases and "does it behave as predicted?".   Run: python test_dqn.py
Each check prints PASS or FAIL and the evidence. Checks in sections A to D need no trained model;
section E uses the saved results of the real training runs (runs/ and results/)."""
import copy
import os
import numpy as np
import torch
from scipy import stats

import dqn
from dqn import (MLP, ReplayBuffer, Game, FrameStack, td_targets, learn, sync_target, epsilon_at, discounted_returns,
                 preprocess_observation, greedy_action, evaluate, moving_average, CONFIGS)

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
results = []
evidence_by_id = {}


def check(name, ok, evidence=""):
    results.append(bool(ok))
    evidence_by_id[name.split(".")[0]] = evidence
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ->  {evidence}" if evidence else ""))


# =====================================================================================
print("=== A. THE LEARNING STEP, WORKED BY HAND ===")
torch.manual_seed(0)
online, target = MLP(4, 2, 8), MLP(4, 2, 8)          # two different small networks
s2 = torch.randn(3, 4)
r = torch.tensor([1.0, 0.0, 5.0])
terminal = torch.tensor([0.0, 0.0, 1.0])              # the third step ended the game
gamma = 0.9
with torch.no_grad():
    q_t, q_o = target(s2), online(s2)
by_hand = []
for i in range(3):
    best_in_target = q_t[i].max().item()
    chosen_by_online = q_t[i, q_o[i].argmax()].item()
    by_hand.append((r[i].item() + gamma * (1 - terminal[i].item()) * best_in_target,
                    r[i].item() + gamma * (1 - terminal[i].item()) * chosen_by_online))
plain = td_targets(online, target, r, s2, terminal, gamma, double=False).numpy()
double = td_targets(online, target, r, s2, terminal, gamma, double=True).numpy()
check("A1. the number each Q value is pulled towards = reward + discount x value of the next state (hand-worked for 3 steps, "
      "the last one a game end, with and without Double DQN)",
      np.allclose(plain, [b[0] for b in by_hand], atol=1e-6) and np.allclose(double, [b[1] for b in by_hand], atol=1e-6),
      f"by hand {np.round([b[0] for b in by_hand], 3)} (plain), {np.round([b[1] for b in by_hand], 3)} (double); "
      f"code gives {np.round(plain, 3)}, {np.round(double, 3)}; the game-end step is exactly its reward, 5")

buf_run = dqn.train("cartpole", seed=0, episodes=3, learning_starts=10 ** 9, buffer=1000, return_buffer=True)[2]
first = [(buf_run.s[i], buf_run.a[i], buf_run.r[i], buf_run.terminal[i]) for i in range(10)]
chain_ok = all(np.allclose(buf_run.s2[i], buf_run.s[i + 1]) for i in range(9))
check("A2. the first 10 steps of a real game: start state close to 0, reward 1 every step, the next state of each step is the state of the "
      "following step, actions are only 0 or 1",
      np.abs(first[0][0]).max() < 0.06 and all(f[2] == 1.0 for f in first) and chain_ok and set(buf_run.a[:10]) <= {0, 1},
      "first state " + str(np.round(first[0][0], 3)) + ", actions " + str(buf_run.a[:10].tolist()) + ", terminal " + str([int(f[3]) for f in first]))

lengths = dqn.train("cartpole", seed=1, episodes=60, learning_starts=10 ** 9, buffer=5000, max_steps=30, return_buffer=True)
log_t, buf_t = lengths[1], lengths[2]
ends = np.cumsum(log_t["length"]) - 1
last_states = buf_t.s2[ends]                                   # the final state of every game
fell = (np.abs(last_states[:, 0]) > 2.4) | (np.abs(last_states[:, 2]) > 0.2095)   # did the cart or the pole really leave the allowed range?
flags = buf_t.terminal[ends]
check("A3. time limit versus game end: a game stopped only by the time limit is NOT marked as ended (the future still counts); "
      "a game where the pole really fell IS marked as ended",
      np.array_equal(flags == 1.0, fell) and (~fell).any() and fell.any(),
      f"{int((~fell).sum())} games stopped by the limit alone (flag 0), {int(fell.sum())} ended by a fall (flag 1)")

# =====================================================================================
print("=== B. THE TARGET NETWORK ===")
torch.manual_seed(1)
net = MLP(4, 2, 16)
tgt = copy.deepcopy(net)
for p in tgt.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
rng = np.random.default_rng(0)
data = (rng.normal(size=(64, 4)).astype(np.float32), rng.integers(0, 2, 64), rng.normal(size=64).astype(np.float32),
        rng.normal(size=(64, 4)).astype(np.float32), np.zeros(64, np.float32))
same_at_start = all(torch.equal(a, b) for a, b in zip(net.parameters(), tgt.parameters()))
before = [p.clone() for p in tgt.parameters()]
for _ in range(25):
    learn(net, tgt, opt, data, 0.9, True, 10.0, "cpu")
target_unchanged = all(torch.equal(a, b) for a, b in zip(before, tgt.parameters()))
online_moved = not all(torch.equal(a, b) for a, b in zip(net.parameters(), tgt.parameters()))
sync_target(net, tgt)
same_after_sync = all(torch.equal(a, b) for a, b in zip(net.parameters(), tgt.parameters()))
check("B1. the target network stays frozen while the online network learns (25 steps), and equals it again after a sync",
      same_at_start and target_unchanged and online_moved and same_after_sync,
      "identical at the start; after 25 learning steps target unchanged, online changed; identical after the sync")
no_grad = all(p.grad is None for p in tgt.parameters())
check("B2. the target network is never trained directly (no gradients, and the optimizer holds only the online network)",
      no_grad and not any(p.requires_grad for p in tgt.parameters()) and
      {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in net.parameters()})
soft = copy.deepcopy(tgt)
w0 = soft.net[0].weight.clone()
sync_target(net, soft, tau=0.1)
expected = 0.9 * w0 + 0.1 * net.net[0].weight
check("B3. a soft update moves the target network only a fraction (tau = 0.1) of the way towards the online network",
      torch.allclose(soft.net[0].weight, expected.detach(), atol=1e-6))

# =====================================================================================
print("=== C. THE REPLAY BUFFER AND THE PICTURES ===")
buf = ReplayBuffer(10, (1,), np.float32)
for i in range(3):
    buf.add(np.array([i]), 0, 0.0, np.array([i]), 0)
few = np.concatenate([buf.sample(50, np.random.default_rng(k))[0] for k in range(20)])
only_filled = few.max() < 3
for i in range(3, 25):
    buf.add(np.array([i]), 0, 0.0, np.array([i]), 0)
kept = sorted(buf.s[:, 0].astype(int).tolist())
check("C1. the buffer: with 3 stored steps only those 3 are ever handed out; after 25 steps in a buffer of 10 exactly the newest 10 remain",
      only_filled and kept == list(range(15, 25)) and buf.size == 10, f"kept {kept[0]} to {kept[-1]}")

big = ReplayBuffer(5000, (1,), np.float32)
for i in range(5000):
    big.add(np.array([i]), 0, 0.0, np.array([i]), 0)
ages = 5000 - 1 - big.born[big.sample(20000, np.random.default_rng(0))[0]]
ks = stats.kstest(ages / 4999.0, "uniform")
check("C2. batches are drawn evenly from the whole buffer (the age of the drawn steps is spread evenly from newest to oldest)",
      ks.pvalue > 0.01 and abs(ages.mean() - 2500) < 80, f"average age {ages.mean():.0f} of 5000 (even spread gives 2500); KS test p = {ks.pvalue:.2f}")

g = Game("pacman", seed=0)
raw_obs, _ = g.env.reset(seed=0)
raw_copy = raw_obs.copy()
cleaned = preprocess_observation(raw_obs)
first_state = g.reset()
nxt, _, _, _ = g.step(1)
second, _, _, _ = g.step(2)
g.reset()
new_game_state = g.reset()
fresh_frames_equal = all(np.array_equal(new_game_state[0], new_game_state[k]) for k in range(1, 4))
check("C3. Pacman pictures: cleaned frame is 88 x 80 with values from -128 to 127 and the game's own picture is left alone; "
      "a new game starts with the first frame repeated 4 times (nothing left over from the old game); each step slides the stack by one frame",
      cleaned.shape == (88, 80, 1) and cleaned.min() >= -128 and cleaned.max() <= 127 and np.array_equal(raw_obs, raw_copy)
      and all(np.array_equal(first_state[0], first_state[k]) for k in range(1, 4)) and fresh_frames_equal
      and np.array_equal(second[:3], nxt[1:]) and g.n_actions == 9 and nxt.dtype == np.int8,
      f"state shape {first_state.shape}, 9 actions")

pm = dqn.train("pacman", seed=0, episodes=2, learning_starts=10 ** 9, buffer=3000, return_buffer=True)
pm_log, pm_buf = pm[1], pm[2]
stored = pm_buf.r[:pm_buf.size]
check("C4. Pacman rewards are squashed to between -1 and +1 for learning (a pellet is 10 points, a ghost 200+), but the scores that are "
      "reported are the real ones",
      stored.max() <= 1.0 and stored.min() >= -1.0 and stored.max() == 1.0 and all(x >= 10 and x % 10 == 0 for x in pm_log["reward"]),
      f"largest stored reward {stored.max():.0f}; real scores of the two games {pm_log['reward']}")

# =====================================================================================
print("=== D. DOES IT BEHAVE AS PREDICTED? (small problems with known answers) ===")


class Chain:
    """A tiny problem solved exactly: 5 squares in a row, start on the left; action 1 = step right, 0 = step left
    (staying put at the left end). Reaching the right end gives reward 1 and ends the game. The state is a one-hot vector."""
    n, gamma = 5, 0.9

    def onehot(self, s):
        v = np.zeros(self.n, np.float32); v[s] = 1
        return v

    def step(self, s, a):
        s2 = min(s + 1, self.n - 1) if a == 1 else max(s - 1, 0)
        return s2, (1.0 if s2 == self.n - 1 else 0.0), s2 == self.n - 1

    def exact_q(self):
        q = np.zeros((self.n, 2))
        for _ in range(500):
            for s in range(self.n - 1):
                for a in range(2):
                    s2, r, done = self.step(s, a)
                    q[s, a] = r + (0 if done else self.gamma * q[s2].max())
        return q


def fit_chain(gamma=None, sync_every=50, ignore_terminal=False, steps=3000, seed=0):
    ch = Chain()
    gamma = ch.gamma if gamma is None else gamma
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    buf = ReplayBuffer(5000, (ch.n,), np.float32)
    for _ in range(5000):                              # random play fills the buffer
        s = int(rng.integers(0, ch.n - 1)); a = int(rng.integers(0, 2))
        s2, r, done = ch.step(s, a)
        buf.add(ch.onehot(s), a, r, ch.onehot(s2), 0.0 if ignore_terminal else float(done))
    net = MLP(ch.n, 2, 32); tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        learn(net, tgt, opt, buf.sample(64, rng)[1:], gamma, True, 10.0, "cpu")
        if sync_every and step % sync_every == 0:
            sync_target(net, tgt)
    with torch.no_grad():
        q = net(torch.tensor(np.stack([ch.onehot(s) for s in range(ch.n - 1)]))).numpy()
    return q, ch.exact_q()[: ch.n - 1]


q, exact = fit_chain()
err = np.abs(q - exact).max()
check("D1. a tiny problem with a known answer (5-square chain): the learned Q values match the exactly solved ones",
      err < 0.06, f"largest difference {err:.3f}; exact Q(start, right) = {exact[0, 1]:.3f}, learned {q[0, 1]:.3f}")

q_frozen, _ = fit_chain(sync_every=0)
q_gamma1, _ = fit_chain(gamma=1.0)
err_frozen, err_gamma1 = np.abs(q_frozen - exact).max(), np.abs(q_gamma1 - exact).max()
check("D2. can D1 fail? Two broken versions must be caught: a target network that is never updated, and a discount of 1 instead of 0.9",
      err_frozen > 5 * err and err_gamma1 > 5 * err and err_frozen > 0.2 and err_gamma1 > 0.2,
      f"error {err:.3f} when correct, {err_frozen:.3f} with a frozen target, {err_gamma1:.3f} with discount 1")

# discount 0: the value of a CartPole state is just the next reward (always 1)
rng = np.random.default_rng(0)
torch.manual_seed(0)
env = Game("cartpole", seed=0)
buf = ReplayBuffer(5000, (4,), np.float32)
state = env.reset()
for _ in range(5000):
    a = int(rng.integers(0, 2)); s2, r, term, trunc = env.step(a)
    buf.add(state, a, r, s2, term)
    state = env.reset() if (term or trunc) else s2
net0 = MLP(4, 2, 32); tgt0 = copy.deepcopy(net0); opt0 = torch.optim.Adam(net0.parameters(), lr=3e-3)
for step in range(1, 1501):
    learn(net0, tgt0, opt0, buf.sample(64, rng)[1:], 0.0, True, 10.0, "cpu")
    if step % 50 == 0:
        sync_target(net0, tgt0)
with torch.no_grad():
    q_all = net0(torch.tensor(buf.s[:2000])).numpy()
check("D3. change the problem: with discount 0 the value of any CartPole state is just the next reward, 1",
      np.abs(q_all - 1.0).max() < 0.1, f"all {q_all.size} Q values between {q_all.min():.3f} and {q_all.max():.3f}")

check("D4. exploration schedule: random-action chance falls in a straight line, 1.0 at the start, halfway at the middle, and stays at the minimum",
      np.isclose(epsilon_at(0, 1.0, 0.05, 1000), 1.0) and np.isclose(epsilon_at(500, 1.0, 0.05, 1000), 0.525)
      and np.isclose(epsilon_at(1000, 1.0, 0.05, 1000), 0.05) and np.isclose(epsilon_at(10 ** 6, 1.0, 0.05, 1000), 0.05))

check("D5. discounted returns, worked by hand: rewards 1, 1, 1 with discount 0.5 give 1.75, 1.5, 1; a value of 4 carried past the end adds 0.5 ** k x 4",
      np.allclose(discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0]) and np.allclose(discounted_returns([1, 1, 1], 0.5, bootstrap=4.0), [2.25, 2.5, 3.0]))

# reduce the randomness: the same seed must give exactly the same training
la = dqn.train("cartpole", seed=3, episodes=25, learning_starts=200)[1]
lb = dqn.train("cartpole", seed=3, episodes=25, learning_starts=200)[1]
lc = dqn.train("cartpole", seed=4, episodes=25, learning_starts=200)[1]
check("D6. the same seed gives an identical training run (rewards, Q values and losses), a different seed a different one",
      all(np.array_equal(la[k], lb[k], equal_nan=True) for k in ("reward", "max_q", "loss")) and not np.array_equal(la["reward"], lc["reward"]),
      "two runs with seed 3 identical; seed 4 differs")

# =====================================================================================
print("=== E. THE REAL TRAINING RUNS ===")
RUNS = os.path.join(HERE, "runs")


def have(*paths):
    return all(os.path.exists(os.path.join(HERE, p)) for p in paths)


def load_net(game, path):
    cfg = CONFIGS[game]
    n_actions = 2 if game == "cartpole" else 9
    net = dqn.make_network(game, n_actions, cfg["hidden"])
    net.load_state_dict(torch.load(os.path.join(HERE, path), map_location="cpu"))
    net.eval()
    return net


exec(open(os.path.join(HERE, "test_results.py")).read()) if os.path.exists(os.path.join(HERE, "test_results.py")) else None

import json
json.dump(evidence_by_id, open(os.path.join(HERE, "test_evidence.json"), "w"), indent=1)   # the report reads its numbers from here
print(f"\n{sum(results)} of {len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
