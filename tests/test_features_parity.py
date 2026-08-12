"""Parity of ProteinFeaturesLigand (edges + ligand context) vs the torch reference.

Instantiates the real torch class (random init), ports it with from_torch, and checks
every output tensor agrees on synthetic input.
"""

import sys

import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, "/tmp")
import ligmpnn_model as ref  # noqa: E402

from jligandmpnn.features import ProteinFeaturesLigand  # noqa: E402

B, L, M, AC, NRBF = 1, 20, 25, 16, 16


def test_features_parity():
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    m = ref.ProteinFeaturesLigand(
        edge_features=128, node_features=128, num_positional_embeddings=16,
        num_rbf=NRBF, top_k=30, augment_eps=0.0, atom_context_num=AC, use_side_chains=False)
    m.eval()

    X = (rng.randn(B, L, 4, 3) * 5).astype(np.float32)
    mask = np.ones((B, L), np.float32)
    Y = (rng.randn(B, L, M, 3) * 5).astype(np.float32)
    Y_t = rng.randint(1, 30, size=(B, L, M)).astype(np.float32)
    Y_m = (rng.rand(B, L, M) > 0.2).astype(np.float32)
    R_idx = np.tile(np.arange(L), (B, 1)).astype(np.float32)
    chain_labels = np.zeros((B, L), np.float32)

    fd = {k: torch.tensor(v) for k, v in dict(
        X=X, mask=mask, Y=Y, Y_m=Y_m, Y_t=Y_t, R_idx=R_idx, chain_labels=chain_labels).items()}
    with torch.no_grad():
        Vt, Et, Eit, Ynt, Yet, Ymt = m(fd)

    j = ProteinFeaturesLigand.from_torch(m)
    Vj, Ej, Eij, Ynj, Yej, Ymj = j(
        *[jnp.asarray(a) for a in (X, mask, Y, Y_m, Y_t, R_idx, chain_labels)])

    assert np.array_equal(np.asarray(Eij), Eit.numpy()), "E_idx mismatch"
    for name, jt, tt in [("V", Vj, Vt), ("E", Ej, Et), ("Y_nodes", Ynj, Ynt),
                         ("Y_edges", Yej, Yet), ("Y_m", Ymj, Ymt)]:
        err = np.abs(np.asarray(jt) - tt.numpy()).max()
        assert err < 1e-4, f"{name} mismatch: {err}"
    print("FEATURES PARITY OK: E_idx exact; V/E/Y_nodes/Y_edges/Y_m < 1e-4")


if __name__ == "__main__":
    test_features_parity()
