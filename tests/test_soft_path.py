"""score_soft (differentiable) must agree with score (integer) on one-hot input, and its
gradient wrt the soft sequence must flow. Uses random init -- no checkpoint needed."""


import jax
import jax.numpy as jnp
import ligmpnn_model as ref
import numpy as np
import torch

from jligandmpnn.model import LigandMPNN

B, L, M = 1, 18, 20


def _inputs():
    rng = np.random.RandomState(3)
    return {
        "X": (rng.randn(B, L, 4, 3) * 5).astype(np.float32), "mask": np.ones((B, L), np.float32),
        "Y": (rng.randn(B, L, M, 3) * 5).astype(np.float32),
        "Y_t": rng.randint(1, 30, (B, L, M)).astype(np.float32),
        "Y_m": (rng.rand(B, L, M) > 0.2).astype(np.float32),
        "R_idx": np.tile(np.arange(L), (B, 1)).astype(np.float32),
        "chain_labels": np.zeros((B, L), np.float32),
        "S": rng.randint(0, 20, (B, L)),  # standard AAs only (0..19)
        "chain_mask": np.ones((B, L), np.float32), "randn": rng.randn(B, L).astype(np.float32)}


def test_soft_matches_hard():
    torch.manual_seed(0)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=16, atom_context_num=M)
    m.eval()
    j = LigandMPNN.from_torch(m)
    a = {k: jnp.asarray(v) for k, v in _inputs().items()}
    order = ("X", "mask", "Y", "Y_m", "Y_t", "R_idx", "chain_labels", "chain_mask", "randn")

    lp_hard = j.score(a["S"], *[a[k] for k in order])
    soft = jax.nn.one_hot(a["S"], 20)
    lp_soft = j.score_soft(soft, *[a[k] for k in order])
    err = np.abs(np.asarray(lp_soft) - np.asarray(lp_hard)).max()
    assert err < 1e-5, f"soft vs hard mismatch: {err}"

    # gradient wrt soft sequence flows
    Soh = jax.nn.one_hot(a["S"], 21)
    def nll(soft):
        return -(Soh * j.score_soft(soft, *[a[k] for k in order])).sum()
    g = jax.grad(nll)(soft)
    assert bool(jnp.isfinite(g).all()) and bool((g != 0).any())
    print(f"SOFT PATH OK: soft==hard ({err:.1e}); grad wrt sequence flows (|g|max {jnp.abs(g).max():.2e})")


if __name__ == "__main__":
    test_soft_matches_hard()
