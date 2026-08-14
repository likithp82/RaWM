# RaWM: Regime-Aware World Model

**When Doing Nothing Wins: Motion Regime Imbalance in In-Cabin Driver Pose Forecasting**  
Likith Prabhu — Mercedes-Benz Research and Development India  
*ECCV 2026 Workshop on Explainable and Safe Autonomous Driving (DriveX)*

---

## Key Finding

In naturalistic in-cabin driving datasets, ~90% of clips show a near-stationary driver. Any model trained with standard ERM is biased toward over-predicting motion — making the trivial **zero-velocity (ZV) baseline 34% better** than the state-of-the-art Driver-WM on aggregate MPJPE (53.42 vs 71.47 px).

We show this failure is **structural** (a provable consequence of ERM on class-imbalanced data), introduce the **Velocity Dominance Score (VDS)** to diagnose it, and propose a lightweight **regime gate** (<5K params) as the fix.

## Results

| Model | All-MPJPE ↓ | HM-MPJPE ↓ | VDS ↓ |
|---|---|---|---|
| Zero-Velocity (ZV) | 53.42 | 140.63 | — |
| Driver-WM | 71.47 | 138.03 | 0.883 |
| RaWM v1 (ours) | 53.04 | 130.92 | 0.593 |
| **RaWM v6 (ours)** | **51.05 ± 0.43** | **130.38 ± 0.93** | **0.332** |

VDS > 0.72 across all tested architectures (193M VLM-conditioned, 1M MLP, 200K LSTM) and two datasets (AIDE, Drive&Act). Every architecture with the regime gate attached passes the VDS test (VDS < 0.5).

## Method

RaWM adds three components to Driver-WM:
1. **Regime gate** — 3 kinematic statistics → MLP → routing score *s* ∈ (0,1)
2. **ZV-anchored static branch** — decodes last observed latent, pulled toward ZV by suppression loss
3. **HM loss reweighting** — 5× upweight on high-motion clips in the rollout branch

Blended prediction: `Ŝ = (1−s)·Ŝ_static + s·Ŝ_rollout`

## Velocity Dominance Score (VDS)

VDS(M, D) = fraction of clips where ZV beats model M in per-clip MPJPE.  
A model **passes the VDS test** if VDS < 0.5 (beats ZV on the majority of clips).

## Setup

```bash
# Clone and install
git clone https://github.com/likithp82/RaWM.git
cd RaWM
pip install -r requirements.txt

# Train RaWM v6
python train.py --config configs/rawm_v6.yaml --seed 42

# Evaluate VDS
python evaluate.py --checkpoint outputs/rawm_v6/seed42/best.pt
```

*(Full dataset setup and pretrained weights coming soon.)*

## Citation

```bibtex
@inproceedings{prabhu2026rawm,
  title     = {When Doing Nothing Wins: Motion Regime Imbalance in In-Cabin Driver Pose Forecasting},
  author    = {Prabhu, Likith},
  booktitle = {ECCV 2026 Workshop on Explainable and Safe Autonomous Driving (DriveX)},
  year      = {2026}
}
```
