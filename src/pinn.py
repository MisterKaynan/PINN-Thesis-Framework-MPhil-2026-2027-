#!/usr/bin/env python3
"""
Reduced SIR Physics-Informed Neural Network — PyTorch reimplementation.

Reimplements Millevoi, Pasetto & Ferronato (2024), PLOS Comput Biol 20(9):e1012387.
The reference implementation (github.com/cmillevoi/EpiPINN) uses SciANN 0.6.6.1
on TensorFlow 2.5.3 / Python 3.8, a stack that no longer installs cleanly. We
reimplement rather than resurrect it, and validate against the paper's
published error tables.

MODEL (paper Eq 12-14)
    Reduced SIR:   dI/dt = delta * (R_t - 1) * I
                   dS/dt = -delta * R_t * I
    R(t) recovered by consistency: R = N - S - I. No initial condition is
    imposed: with the reduced form, I is pinned by the data and the ODE
    calibrates R_t (paper Sec 2.3). This matters for Ghana, where the true
    outbreak start is unknown.

    Scaled (Eq 13), with t_s = (t - t0)/(tf - t0) in [0, 1]:
        dI_s/dt_s = delta * (tf - t0) * (R_t - 1) * I_s

LOSSES
    L_D     = mean[ (I_s(t_j) - I_obs_j)^2 ]                     (Eq 7, w_D = 1)
    L_rODE  = mean[ (dI_s/dt_s - delta*(tf-t0)*(R_t-1)*I_s)^2 ]  (Eq 14)
    joint   : L_D + L_rODE, trained simultaneously               (Eq 15)
    split   : stage 1 fits I_s on data alone; stage 2 fits R_t on the
              residual with I_s frozen                           (Eq 16)

    Because L_D and L_rODE are of consistent magnitude in the reduced model,
    no loss-balancing weights are needed (paper Sec 2.3). This is why we can
    drop SciANN's NTK adaptive weighting without penalty.

DELIBERATE DEVIATIONS FROM THE REFERENCE (declare these in the report)
    - Full-batch Adam by default. The paper mini-batches (size 100) over
      N_D + N_C ~ 6100 points, i.e. ~62 steps/epoch. Full-batch is far cheaper
      on limited hardware; --batch-size restores mini-batching.
    - Non-negativity via a squared output activation, matching the reference's
      `output_activation='square'` hard constraint (paper Sec 2.2).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    """Feedforward net with tanh hidden activations and a squared output.

    The squared output enforces non-negativity as a hard architectural
    constraint rather than a penalty term. The paper reports this is more
    effective and more robust than a weak penalty (Sec 2.2).
    """

    def __init__(self, hidden: int, layers: int = 4):
        super().__init__()
        dims = [1] + [hidden] * layers + [1]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        self.apply(self._glorot)

    @staticmethod
    def _glorot(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t) ** 2


@dataclass
class Config:
    delta: float = 0.2          # 1/5 d^-1, infectious period D = 5 (paper Sec 2.5)
    n_collocation: int = 6000
    hidden_I: int = 50          # I network: 4 x 50
    hidden_R: int = 100         # R_t network: 4 x 100
    layers: int = 4
    lr: float = 1e-3
    epochs_joint: int = 5000
    epochs_split_data: int = 3000
    epochs_split_ode: int = 1000
    batch_size: int = 100       # collocation/joint batch (paper Sec 2.6)
    batch_size_data: int = 10   # split stage-1 data batch (paper Sec 2.6)
    plateau_patience: int = 200
    plateau_factor: float = 0.5
    seed: int = 34              # reference notebook uses 34
    device: str = "auto"        # "auto" | "cpu" | "cuda"


class ReducedSIRPINN:
    """Joint and split training for the reduced SIR PINN."""

    def __init__(self, cfg: Config, tf: float, scale_I: float):
        self.cfg = cfg
        self.tf = float(tf)          # horizon in days; t_s = t / tf
        self.scale_I = float(scale_I)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.I_net = MLP(cfg.hidden_I, cfg.layers).to(self.device)
        self.R_net = MLP(cfg.hidden_R, cfg.layers).to(self.device)
        self.history: list[dict] = []

    # -- residual ---------------------------------------------------------- #
    def _residual(self, t_c: torch.Tensor) -> torch.Tensor:
        """dI_s/dt_s - delta*tf*(R_t - 1)*I_s, evaluated at collocation points."""
        t_c = t_c.clone().requires_grad_(True)
        I = self.I_net(t_c)
        dI = torch.autograd.grad(I, t_c, torch.ones_like(I), create_graph=True)[0]
        R = self.R_net(t_c)
        return dI - self.cfg.delta * self.tf * (R - 1.0) * I

    def _collocation(self) -> torch.Tensor:
        pts = np.random.uniform(0.0, 1.0, self.cfg.n_collocation - 1)
        pts = np.insert(pts, 0, 0.0)          # anchor at t_s = 0, as in the reference
        return torch.tensor(pts, dtype=torch.float32).reshape(-1, 1).to(self.device)

    # -- training ---------------------------------------------------------- #
    def _run(self, params, closure, epochs, tag, log_every=500):
        """Full-batch loop. Used for split stage 1, where the data set is tiny."""
        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.cfg.plateau_factor,
            patience=self.cfg.plateau_patience,
        )
        for ep in range(epochs):
            opt.zero_grad()
            loss = closure()
            loss.backward()
            opt.step()
            sched.step(loss.detach())
            if ep % log_every == 0 or ep == epochs - 1:
                self.history.append({"stage": tag, "epoch": ep,
                                     "loss": float(loss.detach())})
        return float(loss.detach())

    def _run_minibatch(self, params, t_data, I_obs, t_c, epochs, tag,
                       use_data=True, use_ode=True, log_every=500,
                       batch_override=None):
        """
        Mini-batch loop, following the reference implementation.

        Data and collocation points are pooled and shuffled together, then
        consumed in batches. Each batch contributes a data term over whichever
        data points it contains and a residual term over its collocation points.

        This matters more than it looks: with ~6,100 pooled points and batch
        size 100, one epoch yields ~62 gradient updates for the same forward
        work as a single full-batch step. The joint objective does not converge
        without it, which is consistent with the paper's report of slow,
        strongly oscillating joint convergence (Figs 3b, 6).
        """
        opt = torch.optim.Adam(params, lr=self.cfg.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.cfg.plateau_factor,
            patience=self.cfg.plateau_patience,
        )
        bs = batch_override or self.cfg.batch_size or 100
        n_d = t_data.shape[0]
        pooled = torch.cat([t_data, t_c], dim=0)
        is_data = torch.zeros(pooled.shape[0], dtype=torch.bool)
        is_data[:n_d] = True
        targets = torch.cat([I_obs, torch.zeros_like(t_c)], dim=0)

        for ep in range(epochs):
            perm = torch.randperm(pooled.shape[0])
            last = 0.0
            for k in range(0, pooled.shape[0], bs):
                idx = perm[k:k + bs]
                tb, db, yb = pooled[idx], is_data[idx], targets[idx]
                opt.zero_grad()
                loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if use_data and db.any():
                    loss = loss + ((self.I_net(tb[db]) - yb[db]) ** 2).mean()
                if use_ode and (~db).any():
                    loss = loss + (self._residual(tb[~db]) ** 2).mean()
                loss.backward()
                opt.step()
                last = float(loss.detach())
            sched.step(last)
            if ep % log_every == 0 or ep == epochs - 1:
                self.history.append({"stage": tag, "epoch": ep, "loss": last})
        return last

    def fit_joint(self, t_data, I_obs):
        """Eq 15: minimise L_D + L_rODE over both networks simultaneously."""
        t_data, I_obs = t_data.to(self.device), I_obs.to(self.device)
        t_c = self._collocation()
        params = list(self.I_net.parameters()) + list(self.R_net.parameters())
        t0 = time.time()
        final = self._run_minibatch(params, t_data, I_obs, t_c,
                                    self.cfg.epochs_joint, "joint")
        return {"mode": "joint", "final_loss": final, "seconds": time.time() - t0}

    def fit_split(self, t_data, I_obs):
        """Eq 16: fit I_s on data alone, then fit R_t on the residual with I_s frozen."""
        t_data, I_obs = t_data.to(self.device), I_obs.to(self.device)
        t_c = self._collocation()
        t0 = time.time()

        # Stage 1 — pure data regression. A plain NN, no physics.
        # Mini-batched at the paper's batch size of 10. This is not cosmetic:
        # full-batch training here gives one gradient update per epoch instead
        # of ~N_D/10, leaving I_s badly under-fitted. Stage 2 then calibrates
        # R_t against that poor fit, which inflates err_Rt and makes split look
        # worse than joint -- the reverse of the published result.
        l1 = self._run_minibatch(list(self.I_net.parameters()),
                                 t_data, I_obs, t_data[:0],
                                 self.cfg.epochs_split_data, "split_data",
                                 use_ode=False, batch_override=self.cfg.batch_size_data)

        # Stage 2 — fully physics-informed regression. I_s is now fixed.
        for p in self.I_net.parameters():
            p.requires_grad_(False)

        l2 = self._run_minibatch(list(self.R_net.parameters()),
                                 t_data[:0], I_obs[:0], t_c,
                                 self.cfg.epochs_split_ode, "split_ode",
                                 use_data=False)

        for p in self.I_net.parameters():
            p.requires_grad_(True)

        return {"mode": "split", "final_loss_data": l1, "final_loss_ode": l2,
                "seconds": time.time() - t0}

    # -- inference --------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, t_s: torch.Tensor):
        t_s = t_s.to(self.device)
        return (self.I_net(t_s).cpu().numpy().ravel() * self.scale_I,
                self.R_net(t_s).cpu().numpy().ravel())


def relative_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Paper Eq 36: 2-norm of the error over the 2-norm of the reference."""
    return float(np.linalg.norm(pred - ref, 2) / np.linalg.norm(ref, 2))


# --------------------------------------------------------------------------- #
# Validation against the paper's published tables
# --------------------------------------------------------------------------- #

def validate_case4(repo: Path, cfg: Config, out: Path, modes=("joint", "split")):
    """
    Case 4 (paper Table 4): reduced model, synthetic time-dependent beta,
    infectious data perturbed with 40% Gaussian error, tf = 120 days.

    Published reference values:
        Error I  : joint 1.411e-1   split 1.331e-1
        Error R_t: joint 4.961e-1   split 4.744e-1
    """
    import pandas as pd

    df = pd.read_table(repo / "Case4.txt")
    I_true = df["Infectious"].to_numpy(float)
    I_obs = df["I data"].to_numpy(float)
    Rt_true = df["Rt"].to_numpy(float)

    tf = len(df)
    scale_I = 1e5                      # SI = 1e5 in the reference notebook
    t = np.arange(tf, dtype=float) / tf
    t_data = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
    I_obs_sc = torch.tensor(I_obs / scale_I, dtype=torch.float32).reshape(-1, 1)

    published = {"joint": {"I": 1.411e-1, "Rt": 4.961e-1},
                 "split": {"I": 1.331e-1, "Rt": 4.744e-1}}
    results = {}

    for mode in modes:
        m = ReducedSIRPINN(cfg, tf=tf, scale_I=scale_I)
        info = m.fit_joint(t_data, I_obs_sc) if mode == "joint" \
            else m.fit_split(t_data, I_obs_sc)
        I_p, R_p = m.predict(t_data)
        results[mode] = {
            "err_I": relative_error(I_p, I_true),
            "err_Rt": relative_error(R_p, Rt_true),
            "published_err_I": published[mode]["I"],
            "published_err_Rt": published[mode]["Rt"],
            "seconds": round(info["seconds"], 1),
        }
        print(f"[{mode:5s}] err_I={results[mode]['err_I']:.4f} "
              f"(paper {published[mode]['I']:.4f})  "
              f"err_Rt={results[mode]['err_Rt']:.4f} "
              f"(paper {published[mode]['Rt']:.4f})  "
              f"{results[mode]['seconds']}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text())["case4"] if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps({"config": asdict(cfg), "case4": prev}, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("reference"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/validation_case4.json"))
    ap.add_argument("--epochs-joint", type=int, default=5000)
    ap.add_argument("--epochs-split-data", type=int, default=3000)
    ap.add_argument("--epochs-split-ode", type=int, default=1000)
    ap.add_argument("--collocation", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--mode", default="both", choices=["joint", "split", "both"])
    a = ap.parse_args()

    cfg = Config(epochs_joint=a.epochs_joint,
                 epochs_split_data=a.epochs_split_data,
                 epochs_split_ode=a.epochs_split_ode,
                 n_collocation=a.collocation, seed=a.seed)
    modes = ("joint", "split") if a.mode == "both" else (a.mode,)
    validate_case4(a.repo, cfg, a.out, modes)
