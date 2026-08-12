"""Parity of the geometry helpers (kNN, RBF) vs the torch LigandMPNN reference."""

import numpy as np
import torch
import jax.numpy as jnp

from jlig_mpnn.features import knn, rbf, get_rbf

TOP_K, NUM_RBF = 30, 16


# --- torch reference, copied verbatim from LigandMPNN model_utils.py ---
def t_gather_edges(edges, idx):
    neighbors = idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def t_dist(X, mask, top_k, eps=1e-6):
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1.0 - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(
        D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False)
    return D_neighbors, E_idx


def t_rbf(D, num_rbf):
    D_mu = torch.linspace(2.0, 22.0, num_rbf).view([1, 1, 1, -1])
    D_sigma = (22.0 - 2.0) / num_rbf
    return torch.exp(-(((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2))


def t_get_rbf(A, B, E_idx, num_rbf):
    D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
    D_A_B_neighbors = t_gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
    return t_rbf(D_A_B_neighbors, num_rbf)


def test_geom_parity():
    rng = np.random.RandomState(0)
    X = (rng.randn(1, 40, 3) * 10).astype(np.float32)
    mask = np.ones((1, 40), np.float32)

    Dt, Et = t_dist(torch.tensor(X), torch.tensor(mask), TOP_K)
    Dj, Ej = knn(jnp.asarray(X), jnp.asarray(mask), TOP_K)
    assert np.array_equal(np.asarray(Ej), Et.numpy()), "E_idx mismatch"
    assert np.abs(np.asarray(Dj) - Dt.numpy()).max() < 1e-4

    assert np.abs(np.asarray(rbf(Dj, NUM_RBF)) - t_rbf(Dt, NUM_RBF).numpy()).max() < 1e-5

    A = (rng.randn(1, 40, 3) * 10).astype(np.float32)
    grt = t_get_rbf(torch.tensor(A), torch.tensor(A), Et, NUM_RBF).numpy()
    grj = np.asarray(get_rbf(jnp.asarray(A), jnp.asarray(A), Ej, NUM_RBF))
    assert np.abs(grj - grt).max() < 1e-4

    print("GEOM PARITY OK: E_idx exact; D / RBF / get_rbf < 1e-4")


if __name__ == "__main__":
    test_geom_parity()
