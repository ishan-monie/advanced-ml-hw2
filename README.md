# Advanced ML – Assignment 2: Deep Q-Network

GitHub user: **ishan-monie**  |  Repository: **advanced-ml-hw2**

A Deep Q-Network (neural network + replay buffer + target network, with Double DQN and Huber loss) that plays **CartPole-v1** (discount 0.95) and **MsPacman-v5** (discount 0.99, pictures as input).
The submitted report is `report/report.pdf`: the graphs, the edge-case and test tables for each game, and the discussion.

## What is in this repository
| File or folder | What it holds |
|---|---|
| `dqn.py` | The whole Deep Q-Network: replay buffer, networks, target network, training loop, test games |
| `evaluate_models.py` | Plays 500 test games with a saved network (scores and predicted/actual values are saved) |
| `make_plots.py` | All graphs (`figures/`), including the two non-standard ones: the CartPole policy map and the MsPacman predicted-against-actual-value plot |
| `test_dqn.py`, `test_results.py` | The edge-case and behaviour checks. Each check prints PASS or FAIL and the evidence |
| `sweep_cartpole.py`, `diagnose_cartpole.py`, `experiment_discount.py` | CartPole tuning, the reason weak networks fail (the cart leaves the track), and the discount 0.95 against 0.99 experiment (results in the `.csv` files) |
| `runs/` | Training logs (`log.npz`) and chosen networks (`model_best.pt`) of the final runs |
| `results/` | Scores of the 500 test games |
| `figures/` | The graphs used in the report |
| `report/` | The PDF report |

## Requirements
Python 3.12. `pip install -r requirements.txt`. CartPole runs on a laptop CPU. MsPacman needs hours; it was trained on one GPU card (`--device cuda`). The finished runs are included, so the graphs and tests work without training.

## How to run
```
python dqn.py --game cartpole --seed 0 --out runs/cartpole_seed0 --snapshots 10,30,60,100,200,400   # about 3 minutes (seeds 0-4 were used)
python dqn.py --game pacman --seed 0 --device cuda --out runs/pacman_seed0                            # about 3 hours on a GPU (seeds 0 and 1 were used)
python evaluate_models.py --game cartpole --model runs/cartpole_seed0/model_best.pt --out results/cartpole_eval.npz
python evaluate_models.py --game pacman   --model runs/pacman_seed0/model_best.pt   --out results/pacman_eval.npz     # 500 games, about 8 minutes on a CPU
python make_plots.py        # all graphs into figures/
python test_dqn.py          # the checks (about 4 minutes); section E uses the saved results
```
Tuning and extra CartPole experiments (about 10 minutes each): `python sweep_cartpole.py`, `python diagnose_cartpole.py`, `python experiment_discount.py`.

## Notes
- The network that is kept (`model_best.pt`) is chosen without looking at the test games. CartPole: the best of 30 greedy validation games played every 10 training games (fixed starting states). MsPacman: the best 100-game moving average of the training reward.
- The 500 test games use their own random seed (12345). CartPole test games are greedy; MsPacman test games use a 5% chance of a random action.
- Seeds are fixed, but the same seed can give a different run if the number of computer threads changes; CartPole runs use 1 thread. GPU and CPU runs of MsPacman are not bit-identical, so the MsPacman test games were played on the CPU.
- Tuning history: `sweep_cartpole_firstrun.csv` and `sweep_cartpole_secondrun.csv` are earlier rounds (model chosen by the training average); `sweep_cartpole.csv` is the last round (model chosen by validation games).
