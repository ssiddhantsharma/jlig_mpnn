"""Parity of the full LigandMPNN score() (log-probs) vs the torch reference."""

import sys

import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, "/tmp")
import ligmpnn_model as ref  # noqa: E402

from jligandmpnn.model import LigandMPNN  # noqa: E402

B, L, M = 1, 18, 20


def test_model_parity():
    torch.manual_seed(0)
    rng = np.random.RandomState(1)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=16, atom_context_num=M,
                        ligand_mpnn_use_side_chain_context=False)
    m.eval()

    X = (rng.randn(B, L, 4, 3) * 5).astype(np.float32)
    mask = np.ones((B, L), np.float32)
    Y = (rng.randn(B, L, M, 3) * 5).astype(np.float32)
    Y_t = rng.randint(1, 30, size=(B, L, M)).astype(np.float32)
    Y_m = (rng.rand(B, L, M) > 0.2).astype(np.float32)
    R_idx = np.tile(np.arange(L), (B, 1)).astype(np.float32)
    chain_labels = np.zeros((B, L), np.float32)
    S = rng.randint(0, 21, size=(B, L))
    chain_mask = np.ones((B, L), np.float32)
    randn = rng.randn(B, L).astype(np.float32)

    fd = {
        "batch_size": 1, "symmetry_residues": [[]],
        "S": torch.tensor(S), "mask": torch.tensor(mask),
        "chain_mask": torch.tensor(chain_mask), "randn": torch.tensor(randn),
        "X": torch.tensor(X), "Y": torch.tensor(Y), "Y_m": torch.tensor(Y_m),
        "Y_t": torch.tensor(Y_t), "R_idx": torch.tensor(R_idx),
        "chain_labels": torch.tensor(chain_labels),
    }
    with torch.no_grad():
        lp_t = m.score(fd, use_sequence=True)["log_probs"].numpy()

    j = LigandMPNN.from_torch(m)
    lp_j = np.asarray(j.score(
        jnp.asarray(S), jnp.asarray(X), jnp.asarray(mask), jnp.asarray(Y),
        jnp.asarray(Y_m), jnp.asarray(Y_t), jnp.asarray(R_idx),
        jnp.asarray(chain_labels), jnp.asarray(chain_mask), jnp.asarray(randn)))

    err = np.abs(lp_j - lp_t).max()
    assert err < 1e-4, f"log_probs mismatch: {err}"
    print(f"MODEL PARITY OK: log_probs max abs err {err:.2e}")


if __name__ == "__main__":
    test_model_parity()
