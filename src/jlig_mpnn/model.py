"""LigandMPNN model (encode + teacher-forced score), ported from ligmpnn_model.ProteinMPNN.

Only model_type="ligand_mpnn" and the no-symmetry, single-batch score path are ported --
that is the differentiable sequence-log-likelihood we need. Dropout is inference-identity.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from .backend import Embedding, LayerNorm, Linear, from_torch
from .features import ProteinFeaturesLigand
from .layers import DecLayer, DecLayerJ, EncLayer, cat_neighbors_nodes, gather_nodes


class LigandMPNN(eqx.Module):
    features: ProteinFeaturesLigand
    W_v: Linear
    W_c: Linear
    W_nodes_y: Linear
    W_edges_y: Linear
    V_C: Linear
    V_C_norm: LayerNorm
    context_encoder_layers: list
    y_context_encoder_layers: list
    W_e: Linear
    W_s: Embedding
    encoder_layers: list
    decoder_layers: list
    W_out: Linear

    def encode(self, X, mask, Y, Y_m, Y_t, R_idx, chain_labels):
        V, E, E_idx, Y_nodes, Y_edges, Y_m = self.features(
            X, mask, Y, Y_m, Y_t, R_idx, chain_labels)
        h_V = jnp.zeros(E.shape[:2] + (E.shape[-1],))
        h_E = self.W_e(E)
        h_E_context = self.W_v(V)

        mask_attend = gather_nodes(mask[..., None], E_idx)[..., 0]
        mask_attend = mask[..., None] * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)
        for yc, cc in zip(self.y_context_encoder_layers, self.context_encoder_layers):
            Y_nodes = yc(Y_nodes, Y_edges, Y_m, Y_m_edges)
            h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], -1)
            h_V_C = cc(h_V_C, h_E_context_cat, mask, Y_m)

        h_V = h_V + self.V_C_norm(self.V_C(h_V_C))
        return h_V, h_E, E_idx

    def _decode(self, h_S, h_V, h_E, E_idx, mask, chain_mask, randn):
        """Teacher-forced decode given per-residue sequence embeddings h_S. -> [B,L,21]."""
        B, L = mask.shape
        chain_mask = mask * chain_mask
        decoding_order = jnp.argsort((chain_mask + 0.0001) * jnp.abs(randn))
        perm = jax.nn.one_hot(decoding_order, L)
        order_mask_backward = jnp.einsum(
            "ij,biq,bjp->bqp", 1 - jnp.triu(jnp.ones((L, L))), perm, perm)
        mask_attend = jnp.take_along_axis(order_mask_backward, E_idx, axis=2)[..., None]
        mask_1D = mask.reshape(B, L, 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder_fw = mask_fw * cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        return jax.nn.log_softmax(self.W_out(h_V), axis=-1)

    def score(self, S, X, mask, Y, Y_m, Y_t, R_idx, chain_labels, chain_mask, randn):
        """Teacher-forced log-probs of integer sequence S. Returns [B,L,21] log_probs."""
        h_V, h_E, E_idx = self.encode(X, mask, Y, Y_m, Y_t, R_idx, chain_labels)
        return self._decode(self.W_s(S), h_V, h_E, E_idx, mask, chain_mask, randn)

    def score_soft(self, soft20, X, mask, Y, Y_m, Y_t, R_idx, chain_labels, chain_mask, randn):
        """Differentiable log-probs of a soft sequence soft20 [B,L,20] in native MPNN order
        (the first 20 alphabet entries are the standard amino acids). Returns [B,L,21]."""
        h_V, h_E, E_idx = self.encode(X, mask, Y, Y_m, Y_t, R_idx, chain_labels)
        h_S = soft20 @ self.W_s.weight[:20]
        return self._decode(h_S, h_V, h_E, E_idx, mask, chain_mask, randn)

    @staticmethod
    def from_torch(m):
        return LigandMPNN(
            features=ProteinFeaturesLigand.from_torch(m.features),
            W_v=from_torch(m.W_v), W_c=from_torch(m.W_c),
            W_nodes_y=from_torch(m.W_nodes_y), W_edges_y=from_torch(m.W_edges_y),
            V_C=from_torch(m.V_C), V_C_norm=from_torch(m.V_C_norm),
            context_encoder_layers=[DecLayer.from_torch(x) for x in m.context_encoder_layers],
            y_context_encoder_layers=[DecLayerJ.from_torch(x) for x in m.y_context_encoder_layers],
            W_e=from_torch(m.W_e), W_s=from_torch(m.W_s),
            encoder_layers=[EncLayer.from_torch(x) for x in m.encoder_layers],
            decoder_layers=[DecLayer.from_torch(x) for x in m.decoder_layers],
            W_out=from_torch(m.W_out))
