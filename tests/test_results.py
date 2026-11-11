# Section E of test_dqn.py (it is run from there): checks on the saved results of the real training runs.
import csv
import json

# ---------------------------------------------------------------------------------------------- CartPole
if have("runs/cartpole_seed0/log.npz", "runs/cartpole_seed0/model_best.pt", "results/cartpole_eval.npz"):
    print("--- CartPole ---")
    cp_log = dict(np.load(os.path.join(HERE, "runs/cartpole_seed0/log.npz")))
    cp_eval = np.load(os.path.join(HERE, "results/cartpole_eval.npz"))
    cp_net = load_net("cartpole", "runs/cartpole_seed0/model_best.pt")
    scores = cp_eval["scores"]

    check("E1. the 500 saved test games: exactly 500 games, each scoring between 1 and 500",
          len(scores) == 500 and scores.min() >= 1 and scores.max() <= 500,
          f"mean {scores.mean():.1f}, std {scores.std():.1f}, lowest {scores.min():.0f}, {np.mean(scores == 500):.0%} of the games reached 500")

    replay = evaluate(cp_net, "cartpole", episodes=25, seed=12345)
    check("E2. the saved network reproduces the saved test scores exactly (first 25 games replayed)",
          np.array_equal(replay, scores[:25]), f"{int((replay == scores[:25]).sum())} of 25 identical")

    rand_scores = []
    env, rr = Game("cartpole", seed=999), np.random.default_rng(0)
    for _ in range(100):
        env.reset(); done, total = False, 0
        while not done:
            _, r, te, tr = env.step(int(rr.integers(2))); total += r; done = te or tr
        rand_scores.append(total)
    t = stats.ttest_ind(scores, rand_scores, equal_var=False)
    check("E3. the trained network is far better than random play (500 test games against 100 random games, statistical test t above 10)",
          scores.mean() > 10 * np.mean(rand_scores) and t.statistic > 10, f"{scores.mean():.1f} against {np.mean(rand_scores):.1f}, t = {t.statistic:.0f}")

    final_net = load_net("cartpole", "runs/cartpole_seed0/model_last.pt") if have("runs/cartpole_seed0/model_last.pt") else None
    if final_net is not None:
        last_scores = evaluate(final_net, "cartpole", episodes=100, seed=777)
        chosen_scores = evaluate(cp_net, "cartpole", episodes=100, seed=777)
        check("E4. choosing the network by greedy validation games matters: the network at the very end of training plays much worse than the chosen one "
              "(the policy keeps oscillating)", chosen_scores.mean() > 2 * last_scores.mean(),
              f"chosen {chosen_scores.mean():.0f}, last {last_scores.mean():.0f} (100 games each, same starting states)")

    max_q = cp_log["max_q"]
    check("E5. the largest Q value rises while the network learns, and settles at the true ceiling 1/(1-0.95) = 20 (a little above it is over-estimation)",
          np.mean(max_q[:30]) < 2 and 19 < np.mean(max_q[-200:]) < 22,
          f"first 30 games {np.mean(max_q[:30]):.2f}; last 200 games {np.mean(max_q[-200:]):.2f}; highest ever {max_q.max():.2f}")

    ev_pred, ev_real = cp_eval["predicted"], cp_eval["realised"]
    check("E6. predicted against actual value in the test games (a game cut off at 500 steps is credited with the value the pole would keep earning): "
          "on average the network is within 5% of reality, and over-estimates rather than under-estimates",
          abs(ev_pred.mean() - ev_real.mean()) < 0.05 * ev_real.mean() and ev_pred.mean() >= ev_real.mean() - 0.1,
          f"average prediction {ev_pred.mean():.2f}, actual {ev_real.mean():.2f}, highest prediction {ev_pred.max():.2f}")

    seed_means = [np.load(os.path.join(HERE, f"results/cartpole_seed{s}_eval.npz"))["scores"].mean() for s in range(5) if have(f"results/cartpole_seed{s}_eval.npz")]
    check("E7. luck: all 5 training seeds solve the game well (every seed above 300, four of five above 490)",
          len(seed_means) == 5 and min(seed_means) > 300 and sum(m > 490 for m in seed_means) >= 4,
          "test means " + ", ".join(f"{m:.0f}" for m in seed_means))

    if have("experiments/diagnose_cartpole.csv"):
        rows = list(csv.DictReader(open(os.path.join(HERE, "experiments", "diagnose_cartpole.csv"))))
        left, fell = sum(int(r["cart_left_track"]) for r in rows), sum(int(r["pole_fell"]) for r in rows)
        check("E8. why do weak networks fail? Prediction: mostly by the cart drifting off the track, not by the pole falling "
              "(a discount of 0.95 only sees about 20 steps ahead)", left > 3 * fell,
              f"{len(rows)} tuning networks x 100 games: cart left the track {left}, pole fell {fell}")

    if have("experiments/experiment_discount.csv"):
        rows = list(csv.DictReader(open(os.path.join(HERE, "experiments", "experiment_discount.csv"))))
        low = [float(r["final_network_100_test_games"]) for r in rows if float(r["discount"]) == 0.95]
        high = [float(r["final_network_100_test_games"]) for r in rows if float(r["discount"]) == 0.99]
        left95 = sum(int(r["cart_left_track"]) for r in rows if float(r["discount"]) == 0.95)
        left99 = sum(int(r["cart_left_track"]) for r in rows if float(r["discount"]) == 0.99)
        reach95 = sum(int(r["reached_500"]) for r in rows if float(r["discount"]) == 0.95)
        reach99 = sum(int(r["reached_500"]) for r in rows if float(r["discount"]) == 0.99)
        check("E9. change the problem: with discount 0.99 instead of 0.95 (a longer view) fewer games should be lost by the cart leaving the track "
              "and some should reach 500. The stronger prediction made before the run, that the average score would be 1.5 times higher, did NOT hold "
              f"({np.mean(low):.0f} with 0.95, {np.mean(high):.0f} with 0.99: the policy still oscillates)",
              left99 < left95 and reach99 > reach95,
              f"5 seeds x 100 games: cart left the track {left95} -> {left99}, reached 500: {reach95} -> {reach99}")

# ---------------------------------------------------------------------------------------------- MsPacman
if have("runs/pacman_seed0/log.npz", "runs/pacman_seed0/model_best.pt", "results/pacman_eval.npz"):
    print("--- MsPacman ---")
    pm_log = dict(np.load(os.path.join(HERE, "runs/pacman_seed0/log.npz")))
    pm_eval = np.load(os.path.join(HERE, "results/pacman_eval.npz"))
    pm_net = load_net("pacman", "runs/pacman_seed0/model_best.pt")
    ps = pm_eval["scores"]

    check("P1. the 500 saved test games: exactly 500 games, every score a multiple of 10 (the game pays 10, 50, 100, ... points) and not negative",
          len(ps) == 500 and ps.min() >= 0 and np.all(ps % 10 == 0),
          f"mean {ps.mean():.1f}, std {ps.std():.1f}, lowest {ps.min():.0f}, highest {ps.max():.0f}")

    replay = evaluate(pm_net, "pacman", episodes=3, seed=12345, eps=float(pm_eval["eps"]))
    check("P2. the saved network reproduces the saved test scores exactly (first 3 games replayed on the CPU)",
          np.array_equal(replay, ps[:3]), f"replayed {replay.tolist()}, saved {ps[:3].tolist()}")

    rand_scores = []
    env, rr = Game("pacman", seed=999), np.random.default_rng(0)
    for _ in range(60):
        env.reset(); done, total = False, 0
        while not done:
            _, r, te, tr = env.step(int(rr.integers(9))); total += r; done = te or tr
        rand_scores.append(total)
    t = stats.ttest_ind(ps, rand_scores, equal_var=False)
    check("P3. the trained network is far better than random play (500 test games against 60 random games, statistical test t above 10)",
          ps.mean() > 2 * np.mean(rand_scores) and t.statistic > 10, f"{ps.mean():.0f} against {np.mean(rand_scores):.0f}, t = {t.statistic:.0f}")

    rew = pm_log["reward"]
    t = stats.ttest_ind(rew[-100:], rew[:100], equal_var=False)
    check("P4. learning happened: the last 100 training games score far more than the first 100 (at least 3 times, and not by luck)",
          rew[-100:].mean() > 3 * rew[:100].mean() and t.pvalue < 0.001,
          f"first 100 games {rew[:100].mean():.0f}, last 100 games {rew[-100:].mean():.0f}, p = {t.pvalue:.1e}")

    seed_info = []
    for s in (0, 1):
        if have(f"runs/pacman_seed{s}/log.npz"):
            lg = np.load(os.path.join(HERE, f"runs/pacman_seed{s}/log.npz"))["reward"]
            seed_info.append((lg[:100].mean(), lg[-100:].mean()))
    check("P5. luck: a second training seed also learns (its last 100 games score at least 3 times its first 100)",
          len(seed_info) == 2 and all(b > 3 * a for a, b in seed_info),
          "; ".join(f"seed {i}: {a:.0f} -> {b:.0f}" for i, (a, b) in enumerate(seed_info)))

    eps_log, step_log = pm_log["epsilon"], pm_log["step"]
    check("P6. exploration: the random-action chance starts near 1, reaches its minimum 0.05 only after 400,000 steps, and stays there",
          eps_log[0] > 0.97 and np.isclose(eps_log[-1], 0.05) and np.all(eps_log[step_log < 400_000] > 0.05) and np.all(np.isclose(eps_log[step_log > 400_500], 0.05)),
          f"first game {eps_log[0]:.3f}, last game {eps_log[-1]:.3f}")

    pq, rq = pm_eval["predicted"], pm_eval["realised"]
    corr = np.corrcoef(pq, rq)[0, 1]
    check("P7. predicted against actual value in the test games (rewards clipped to -1..+1, discount 0.99): the predictions follow the actual returns "
          "(correlation above 0.5)", corr > 0.5,
          f"correlation {corr:.2f}; average prediction {pq.mean():.1f}, average actual {rq.mean():.1f}; highest prediction {pq.max():.1f}")

    mq = pm_log["max_q"]
    check("P8. the largest Q value starts near 0, rises while the network learns, and stays far below the ceiling 1/(1-0.99) = 100",
          np.mean(mq[:25]) < 2 and np.mean(mq[-200:]) > 5 * np.mean(mq[:25]) and mq.max() < 100,
          f"first 25 games {np.mean(mq[:25]):.2f}; last 200 games {np.mean(mq[-200:]):.1f}; highest ever {mq.max():.1f}")

    torch.manual_seed(0)
    fresh = dqn.make_network("pacman", 9, CONFIGS["pacman"]["hidden"])
    fresh.eval()
    fresh_scores = evaluate(fresh, "pacman", episodes=20, seed=12345, eps=float(pm_eval["eps"]))
    check("P9. can the results fail? A network with random weights (never trained) should score like random play, far below the trained network",
          fresh_scores.mean() < 0.5 * ps.mean(), f"untrained network {fresh_scores.mean():.0f} (20 games), trained network {ps.mean():.0f}")
