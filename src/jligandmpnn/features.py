"""ProteinFeaturesLigand geometry, ported from LigandMPNN model_utils.py.

Built up block by block (kNN graph -> RBF distances -> edges -> ligand context), each
parity-checked against the torch reference.
"""

import jax
import jax.numpy as jnp

from .layers import gather_edges


def knn(X, mask, top_k, eps=1e-6):
    """kNN graph on coordinates. X [B,L,3], mask [B,L] -> (D_neighbors, E_idx) [B,L,K]."""
    mask_2D = mask[:, :, None] * mask[:, None, :]
    dX = X[:, None, :, :] - X[:, :, None, :]
    D = mask_2D * jnp.sqrt((dX ** 2).sum(-1) + eps)
    D_max = D.max(-1, keepdims=True)
    D_adjust = D + (1.0 - mask_2D) * D_max
    k = min(top_k, X.shape[1])
    neg_D, E_idx = jax.lax.top_k(-D_adjust, k)  # k smallest distances, ascending
    return -neg_D, E_idx


def rbf(D, num_rbf, d_min=2.0, d_max=22.0):
    """Gaussian radial basis expansion of distances D -> [..., num_rbf]."""
    mu = jnp.linspace(d_min, d_max, num_rbf)
    sigma = (d_max - d_min) / num_rbf
    return jnp.exp(-(((D[..., None] - mu) / sigma) ** 2))


def get_rbf(A, B, E_idx, num_rbf):
    """RBF of A_i-B_j distances gathered onto the kNN edges. [B,L,K,num_rbf]."""
    D_A_B = jnp.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + 1e-6)
    D_neighbors = gather_edges(D_A_B[..., None], E_idx)[..., 0]
    return rbf(D_neighbors, num_rbf)
