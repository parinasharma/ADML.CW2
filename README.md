# Predictive Maintenance for a Transport Fleet

Coursework project (University of Hertfordshire, BSc Data Science).
A transport operator runs trains, planes and buses and wants to move from fixed
service schedules to data-driven maintenance. This project has two parts.

- **Part A: Representation learning (trains).** Do train sensor readings show the
  difference between normal operation and a developing fault?
- **Part C: Reinforcement learning (fleet scheduling).** Can an agent learn when to
  service each vehicle so that profit is highest?

## Contents

| File | Description |
|---|---|
| `partA_and_C_FINAL.ipynb` | Full notebook: code, outputs and commentary for Parts A and C |
| `Fleet_Predictive_Maintenance.pptx` | Presentation slides |

## Part A: train sensor data (MetroPT-3)

**Data:** MetroPT-3 air compressor dataset (UCI Machine Learning Repository):
about 1.5 million readings, 15 sensors, 4 real failures (April to July 2020).

**Preparation**
1. Averaged readings to 1-minute values (252,720 minutes).
2. Labelled each minute: normal, pre-failure (24 hours before a failure), or failure.
3. Split normal days into train, validation and test sets (about 80/10/10).
   Failure and pre-failure minutes are never used for fitting.
4. Standardised the sensors using training data only.

**Methods**
- PCA (2 components, plus a sweep over the number of components)
- Linear autoencoder (same 2-D size, 5 seeds, compared with PCA using principal
  angles and Procrustes alignment)
- Self-Organising Map (25x25 grid, k-means on the prototypes)

**Key results**

| Method | AUC: failure vs normal | AUC: pre-failure and failure vs normal |
|---|---|---|
| PCA (2) | 0.959 | 0.730 |
| Linear AE (2) | 0.959 | 0.729 |
| SOM (25x25) | 0.870 | 0.725 |

- Two PCA components explain 61% of the variance.
- The autoencoder plane is within 3.6 and 1.1 degrees of the PCA plane.
- Failures are easy to detect; early warning (24 hours before) is weak.

## Part C: Q-learning for fleet maintenance

**Environment:** custom Gymnasium environment (`FleetMaintenance-v0`): 3 vehicles,
wear levels 0 to 3 (64 states), 4 actions (do nothing or service vehicle 1, 2 or 3),
60-step episodes. Revenue falls as wear rises; servicing costs 4; a breakdown costs 40.
Warm-up on `FrozenLake-v1`.

**Experiments:** four epsilon schedules (greedy, constant 0.1, constant 0.5,
exponential decay), 5 seeds, 15,000 episodes, compared with four baseline rules
(never, fixed, threshold, random). Also tested: adding a time feature to the state,
and the effect of decay speed.

**Key results (test episodes, greedy evaluation)**

| Policy | Mean return |
|---|---|
| Q-learning, decaying epsilon | 366.8 |
| Baseline: threshold rule | 366.7 |
| Q-learning, constant epsilon 0.5 | 363.0 |
| Q-learning, constant epsilon 0.1 | 362.2 |
| Baseline: fixed schedule | 184.2 |
| Q-learning, greedy (no exploration) | 171.6 |
| Baseline: random | 121.8 |
| Baseline: never service | -43.7 |

The decaying-epsilon agent matches the best hand-written rule with zero breakdowns.
Adding a time feature made it worse (64 states: 366.8; 768 states: 319.2).

## Limitations

- Only 4 real failures, so results may reflect individual events.
- Operating mode (for example motor current) can look like a fault.
- The Part C fleet is simulated with assumed costs.
- A Q-table does not scale to large fleets.

## Future work

Sequence models (for example LSTM) for earlier warning, modelling each operating
mode separately, real repair-cost data, and Deep Q-Networks for larger fleets.

## How to run

1. Download `metropt_3_dataset.zip` from the UCI repository (search "MetroPT-3 Dataset")
   and place it next to the notebook or in `./data`, or set the `METROPT_PATH`
   environment variable to its location. The dataset is not included in this repo.
2. Install the requirements:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow minisom gymnasium
   ```
3. Open `partA_and_C_FINAL.ipynb` in Jupyter or Google Colab and run all cells.
   Random seed: 42.

## Data source

MetroPT-3 Dataset, UCI Machine Learning Repository (Metro do Porto air production unit).
