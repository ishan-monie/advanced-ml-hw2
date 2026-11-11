# Advanced ML – Assignment 2: Deep Q-Network

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw2**

A Deep Q-Network (neural network + replay buffer + target network, with Double DQN and Huber loss) that plays **CartPole-v1** (discount 0.95) and **MsPacman-v5** (discount 0.99, pictures as input).

## Where is what

```
advanced-ml-hw2/
├── README.md             this file
├── requirements.txt      Python packages to install
│
├── report/               THE REPORT
│   └── report_v2.pdf       the report (graphs, edge cases, tests, discussion, run instructions)
│
├── src/                  THE CODE
│   ├── dqn.py              the whole Deep Q-Network: replay buffer, networks, target network, training loop, test games
│   ├── evaluate_models.py  plays 500 test games with a saved network
│   └── make_plots.py       makes every graph in figures/ (including the two non-standard ones)
│
├── tests/                THE EDGE-CASE AND BEHAVIOUR CHECKS
│   ├── test_dqn.py         checks that need no trained network (and runs test_results.py at the end)
│   ├── test_results.py     checks on the saved CartPole and MsPacman runs
│
├── figures/              the graphs used in the report (PNG)
│   ├── cartpole_*.png      max Q, training reward, 500-game histogram, seeds, policy map (non-standard)
│   └── pacman_*.png        max Q, training reward, 500-game histogram, seeds, predicted-vs-actual value (non-standard)
│
├── runs/                 TRAINING OUTPUT of the final runs (cartpole_seed0-4, pacman_seed0-1)
│   └── <run>/              log.npz (training log), run_info.json, and model_best.pt (kept network, only for the two main runs cartpole_seed0 and pacman_seed0)
│
├── results/              scores of the 500 test games, as .npz files
│
└── experiments/          EXTRA CARTPOLE EXPERIMENTS (tuning, not needed to run the final result)
    ├── sweep_cartpole.py            compares settings on several seeds
    ├── diagnose_cartpole.py         shows why weak networks fail (the cart leaves the track)
    ├── experiment_discount.py       discount 0.95 against 0.99
    └── *.csv                        their results (sweep_cartpole_firstrun / _secondrun are earlier rounds)
```

## Requirements
Python 3.12. `pip install -r requirements.txt`. CartPole runs on a laptop CPU. MsPacman needs hours; it was trained on one GPU card (`--device cuda`). The finished runs are included, so the graphs and tests work without training.

## How to run
Run everything from the main folder of the repository.
```
python src/dqn.py --game cartpole --seed 0 --out runs/cartpole_seed0 --snapshots 10,30,60,100,200,400   # about 3 minutes (seeds 0-4 were used)
python src/dqn.py --game pacman --seed 0 --device cuda --out runs/pacman_seed0                            # about 3 hours on a GPU (seeds 0 and 1 were used)
python src/evaluate_models.py --game cartpole --model runs/cartpole_seed0/model_best.pt --out results/cartpole_eval.npz
python src/evaluate_models.py --game pacman   --model runs/pacman_seed0/model_best.pt   --out results/pacman_eval.npz     # 500 games, about 8 minutes on a CPU
python src/make_plots.py        # all graphs into figures/
python tests/test_dqn.py        # the checks (about 4 minutes); section E uses the saved results
```
Extra CartPole experiments (about 10 minutes each): `python experiments/sweep_cartpole.py`, `python experiments/diagnose_cartpole.py`, `python experiments/experiment_discount.py`.

## Notes
- The network that is kept (`model_best.pt`) is chosen without looking at the test games. CartPole: the best of 30 greedy validation games played every 10 training games (fixed starting states). MsPacman: the best 100-game moving average of the training reward.
- The 500 test games use their own random seed (12345). CartPole test games are greedy; MsPacman test games use a 5% chance of a random action.
- Seeds are fixed, but the same seed can give a different run if the number of computer threads changes; CartPole runs use 1 thread. GPU and CPU runs of MsPacman are not bit-identical, so the MsPacman test games were played on the CPU.
- Tuning history: `experiments/sweep_cartpole_firstrun.csv` and `_secondrun.csv` are earlier rounds (model chosen by the training average); `sweep_cartpole.csv` is the last round (model chosen by validation games).
