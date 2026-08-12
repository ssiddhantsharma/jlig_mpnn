"""ProteinFeaturesLigand geometry, ported from LigandMPNN model_utils.py.

Built up block by block (kNN graph -> RBF distances -> edges -> ligand context), each
parity-checked against the torch reference.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import LayerNorm, Linear, from_torch
from .layers import PositionalEncodings, gather_edges


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


def _normalize(v, eps=1e-12):
    return v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), eps)


def _make_angle_features(A, B, C, Y):
    """Local frame (e1,e2,e3) at each residue; project ligand atoms Y into it -> [B,L,M,4]."""
    v1, v2 = A - B, C - B
    e1 = _normalize(v1)
    u2 = v2 - e1 * (e1 * v2).sum(-1, keepdims=True)
    e2 = _normalize(u2)
    e3 = jnp.cross(e1, e2)
    R = jnp.stack([e1, e2, e3], axis=-1)  # [B,L,3,3]
    local = jnp.einsum("blqp,blyq->blyp", R, Y - B[:, :, None, :])
    rxy = jnp.sqrt(local[..., 0] ** 2 + local[..., 1] ** 2 + 1e-8)
    rxyz = jnp.linalg.norm(local, axis=-1) + 1e-8
    return jnp.stack(
        [local[..., 0] / rxy, local[..., 1] / rxy, rxy / rxyz, local[..., 2] / rxyz], -1)


def _cb(N, Ca, C):
    b, c = Ca - N, C - Ca
    return -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Ca


# order matches the torch RBF_all list (25 backbone atom pairs)
_RBF_PAIRS = [
    ("N", "N"), ("C", "C"), ("O", "O"), ("Cb", "Cb"), ("Ca", "N"), ("Ca", "C"),
    ("Ca", "O"), ("Ca", "Cb"), ("N", "C"), ("N", "O"), ("N", "Cb"), ("Cb", "C"),
    ("Cb", "O"), ("O", "C"), ("N", "Ca"), ("C", "Ca"), ("O", "Ca"), ("Cb", "Ca"),
    ("C", "N"), ("O", "N"), ("Cb", "N"), ("C", "Cb"), ("O", "Cb"), ("C", "O"),
]


class ProteinFeaturesLigand(eqx.Module):
    """LigandMPNN input featurizer (use_side_chains=False). Returns V, E, E_idx, Y_nodes,
    Y_edges, Y_m. augment_eps noise is training-only and omitted (inference is deterministic)."""

    embeddings: PositionalEncodings
    edge_embedding: Linear
    norm_edges: LayerNorm
    node_project_down: Linear
    norm_nodes: LayerNorm
    type_linear: Linear
    y_nodes: Linear
    y_edges: Linear
    norm_y_edges: LayerNorm
    norm_y_nodes: LayerNorm
    periodic_table_features: jax.Array
    num_rbf: int
    top_k: int

    def __call__(self, X, mask, Y, Y_m, Y_t, R_idx, chain_labels):
        Ca, N, C, O = X[:, :, 1], X[:, :, 0], X[:, :, 2], X[:, :, 3]
        Cb = _cb(N, Ca, C)
        atoms = {"N": N, "Ca": Ca, "C": C, "O": O, "Cb": Cb}

        D_neighbors, E_idx = knn(Ca, mask, self.top_k)
        RBF_all = [rbf(D_neighbors, self.num_rbf)]
        RBF_all += [get_rbf(atoms[a], atoms[b], E_idx, self.num_rbf) for a, b in _RBF_PAIRS]
        RBF_all = jnp.concatenate(RBF_all, -1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[..., None], E_idx)[..., 0]
        d_chains = (chain_labels[:, :, None] - chain_labels[:, None, :] == 0).astype(jnp.int32)
        E_chains = gather_edges(d_chains[..., None], E_idx)[..., 0]
        E_positional = self.embeddings(offset, E_chains)
        E = self.norm_edges(self.edge_embedding(jnp.concatenate([E_positional, RBF_all], -1)))

        # ligand atom context. Y is [B,L,num_context_atoms,3], already limited per residue
        # by the data featurizer; the topk reduction here is the use_side_chains-only path.
        Y_t = Y_t.astype(jnp.int32)
        Y_t_g = self.periodic_table_features[1][Y_t]
        Y_t_p = self.periodic_table_features[2][Y_t]
        Y_t_1hot_ = jnp.concatenate([
            jax.nn.one_hot(Y_t, 120), jax.nn.one_hot(Y_t_g, 19), jax.nn.one_hot(Y_t_p, 8)], -1)
        Y_t_1hot = self.type_linear(Y_t_1hot_)

        D_bb_Y = [get_rbf_pair(a, Y, self.num_rbf) for a in (N, Ca, C, O, Cb)]
        f_angles = _make_angle_features(N, Ca, C, Y)
        D_all = jnp.concatenate(D_bb_Y + [Y_t_1hot, f_angles], -1)
        V = self.norm_nodes(self.node_project_down(D_all))

        Y_Y = jnp.sqrt(((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2).sum(-1) + 1e-6)
        Y_edges = self.norm_y_edges(self.y_edges(rbf(Y_Y, self.num_rbf)))
        Y_nodes = self.norm_y_nodes(self.y_nodes(Y_t_1hot_))
        return V, E, E_idx, Y_nodes, Y_edges, Y_m

    @staticmethod
    def from_torch(m):
        return ProteinFeaturesLigand(
            embeddings=PositionalEncodings.from_torch(m.embeddings),
            edge_embedding=from_torch(m.edge_embedding),
            norm_edges=from_torch(m.norm_edges),
            node_project_down=from_torch(m.node_project_down),
            norm_nodes=from_torch(m.norm_nodes),
            type_linear=from_torch(m.type_linear),
            y_nodes=from_torch(m.y_nodes),
            y_edges=from_torch(m.y_edges),
            norm_y_edges=from_torch(m.norm_y_edges),
            norm_y_nodes=from_torch(m.norm_y_nodes),
            periodic_table_features=from_torch(m.periodic_table_features),
            num_rbf=m.num_rbf, top_k=m.top_k)


def get_rbf_pair(A, Y, num_rbf):
    """RBF of backbone-atom A_i to ligand-atom distances (no kNN gather). [B,L,M,num_rbf]."""
    return rbf(jnp.sqrt(((A[:, :, None, :] - Y) ** 2).sum(-1) + 1e-6), num_rbf)
