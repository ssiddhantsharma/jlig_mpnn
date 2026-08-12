"""Message-passing layers + graph helpers, ported from LigandMPNN model_utils.py.

Dropout is inference-identity and omitted; GELU is exact (approximate=False) to match
torch. from_torch staticmethods are registered against the torch classes by the loader.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import LayerNorm, Linear, from_torch


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def gather_nodes(nodes, E_idx):
    # nodes [B,N,C], E_idx [B,N,K] -> [B,N,K,C]
    return jax.vmap(lambda nd, idx: nd[idx])(nodes, E_idx)


def gather_edges(edges, E_idx):
    # edges [B,N,N,C], E_idx [B,N,K] -> [B,N,K,C]
    idx = jnp.broadcast_to(E_idx[..., None], E_idx.shape + (edges.shape[-1],))
    return jnp.take_along_axis(edges, idx, axis=2)


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    return jnp.concatenate([h_neighbors, gather_nodes(h_nodes, E_idx)], axis=-1)


def _expand_cat(h_V, h_E):
    h_V_expand = jnp.broadcast_to(h_V[..., None, :], h_E.shape[:-1] + (h_V.shape[-1],))
    return jnp.concatenate([h_V_expand, h_E], axis=-1)


class PositionWiseFeedForward(eqx.Module):
    W_in: Linear
    W_out: Linear

    def __call__(self, h):
        return self.W_out(_gelu(self.W_in(h)))

    @staticmethod
    def from_torch(m):
        return PositionWiseFeedForward(W_in=from_torch(m.W_in), W_out=from_torch(m.W_out))


class DecLayer(eqx.Module):
    norm1: LayerNorm
    norm2: LayerNorm
    W1: Linear
    W2: Linear
    W3: Linear
    dense: PositionWiseFeedForward
    scale: float

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_EV = _expand_cat(h_V, h_E)
        h_message = self.W3(_gelu(self.W2(_gelu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = h_message.sum(-2) / self.scale
        h_V = self.norm1(h_V + dh)
        h_V = self.norm2(h_V + self.dense(h_V))
        if mask_V is not None:
            h_V = mask_V[..., None] * h_V
        return h_V

    @staticmethod
    def from_torch(m):
        return DecLayer(
            norm1=from_torch(m.norm1), norm2=from_torch(m.norm2),
            W1=from_torch(m.W1), W2=from_torch(m.W2), W3=from_torch(m.W3),
            dense=from_torch(m.dense), scale=m.scale)


# DecLayerJ is DecLayer applied over one extra (ligand-atom) dimension; the broadcast in
# _expand_cat handles the rank difference, so the computation is identical.
class DecLayerJ(DecLayer):
    @staticmethod
    def from_torch(m):
        return DecLayerJ(
            norm1=from_torch(m.norm1), norm2=from_torch(m.norm2),
            W1=from_torch(m.W1), W2=from_torch(m.W2), W3=from_torch(m.W3),
            dense=from_torch(m.dense), scale=m.scale)


class EncLayer(eqx.Module):
    norm1: LayerNorm
    norm2: LayerNorm
    norm3: LayerNorm
    W1: Linear
    W2: Linear
    W3: Linear
    W11: Linear
    W12: Linear
    W13: Linear
    dense: PositionWiseFeedForward
    scale: float

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = _expand_cat(h_V, cat_neighbors_nodes(h_V, h_E, E_idx))
        h_message = self.W3(_gelu(self.W2(_gelu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = h_message.sum(-2) / self.scale
        h_V = self.norm1(h_V + dh)
        h_V = self.norm2(h_V + self.dense(h_V))
        if mask_V is not None:
            h_V = mask_V[..., None] * h_V

        h_EV = _expand_cat(h_V, cat_neighbors_nodes(h_V, h_E, E_idx))
        h_message = self.W13(_gelu(self.W12(_gelu(self.W11(h_EV)))))
        h_E = self.norm3(h_E + h_message)
        return h_V, h_E

    @staticmethod
    def from_torch(m):
        return EncLayer(
            norm1=from_torch(m.norm1), norm2=from_torch(m.norm2), norm3=from_torch(m.norm3),
            W1=from_torch(m.W1), W2=from_torch(m.W2), W3=from_torch(m.W3),
            W11=from_torch(m.W11), W12=from_torch(m.W12), W13=from_torch(m.W13),
            dense=from_torch(m.dense), scale=m.scale)


class PositionalEncodings(eqx.Module):
    linear: Linear
    max_relative_feature: int

    def __call__(self, offset, mask):
        mrf = self.max_relative_feature
        d = jnp.clip(offset + mrf, 0, 2 * mrf) * mask + (1 - mask) * (2 * mrf + 1)
        d_onehot = jax.nn.one_hot(d.astype(jnp.int32), 2 * mrf + 2)
        return self.linear(d_onehot)

    @staticmethod
    def from_torch(m):
        return PositionalEncodings(
            linear=from_torch(m.linear), max_relative_feature=m.max_relative_feature)
