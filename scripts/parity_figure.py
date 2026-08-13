"""Proof figure: jligandmpnn score() log-probs vs the torch LigandMPNN reference.

Self-contained (uses the vendored reference under ./reference). Run: `python scripts/parity_figure.py`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
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

    diff = lp_j - lp_t
    err = np.abs(diff).max()
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4))

    lim = [lp_t.min() - 0.3, lp_t.max() + 0.3]
    a.plot(lim, lim, color="0.7", lw=0.8, zorder=1)
    a.scatter(lp_t, lp_j, s=7, color="#0072B2", alpha=0.5, linewidths=0, zorder=2)
    a.set(xlabel="torch  log P", ylabel="jligandmpnn  log P", xlim=lim, ylim=lim)
    a.set_aspect("equal")
    a.set_title(f"{L * 21} per-residue log-probs", fontsize=10)

    b.hist(diff, bins=40, color="#0072B2")
    b.axvline(0, color="0.7", lw=0.8)
    b.set(xlabel="jligandmpnn - torch", ylabel="count")
    b.set_title(f"max |Δ| = {err:.0e}", fontsize=10)
    b.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "figures" / "parity.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out, "max abs err", err)


if __name__ == "__main__":
    main()
