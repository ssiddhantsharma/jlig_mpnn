"""Proof figure: jligandmpnn score() log-probs vs the torch LigandMPNN reference.

Needs torch + the LigandMPNN reference module importable as `ligmpnn_model` (LIGMPNN_MODEL_DIR).
Saves figures/parity.png. Run: `LIGMPNN_MODEL_DIR=... python scripts/parity_figure.py`.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.environ["LIGMPNN_MODEL_DIR"])
import ligmpnn_model as ref

from jligandmpnn.model import LigandMPNN

B, L, M = 1, 40, 25


def main():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=16, atom_context_num=M)
    m.eval()

    a = {
        "X": (rng.randn(B, L, 4, 3) * 5).astype(np.float32), "mask": np.ones((B, L), np.float32),
        "Y": (rng.randn(B, L, M, 3) * 5).astype(np.float32),
        "Y_t": rng.randint(1, 30, (B, L, M)).astype(np.float32),
        "Y_m": (rng.rand(B, L, M) > 0.2).astype(np.float32),
        "R_idx": np.tile(np.arange(L), (B, 1)).astype(np.float32),
        "chain_labels": np.zeros((B, L), np.float32), "S": rng.randint(0, 21, (B, L)),
        "chain_mask": np.ones((B, L), np.float32), "randn": rng.randn(B, L).astype(np.float32)}

    fd = {"batch_size": 1, "symmetry_residues": [[]]}
    fd.update({k: torch.tensor(v) for k, v in a.items()})
    with torch.no_grad():
        lp_t = m.score(fd, use_sequence=True)["log_probs"].numpy().reshape(-1)
    j = LigandMPNN.from_torch(m)
    order = ("S", "X", "mask", "Y", "Y_m", "Y_t", "R_idx", "chain_labels", "chain_mask", "randn")
    lp_j = np.asarray(j.score(*[jnp.asarray(a[k]) for k in order])).reshape(-1)

    err = np.abs(lp_j - lp_t).max()
    lim = [lp_t.min() - 0.3, lp_t.max() + 0.3]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(lim, lim, color="0.7", lw=0.8, zorder=1)
    ax.scatter(lp_t, lp_j, s=6, color="#222222", alpha=0.45, linewidths=0, zorder=2)
    ax.set(xlabel="torch  log P", ylabel="jligandmpnn  log P", xlim=lim, ylim=lim)
    ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"max |Δ| {err:.0e}", transform=ax.transAxes, va="top", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "parity.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out, "max abs err", err)


if __name__ == "__main__":
    main()
