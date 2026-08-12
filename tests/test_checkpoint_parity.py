"""Parity of score() against the real trained checkpoint (not just random init).

Skips unless the public checkpoint is present. Fetch it with:
  curl -sL -o /tmp/ligandmpnn_v_32_010_25.pt \\
    https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import torch

CKPT = "/tmp/ligandmpnn_v_32_010_25.pt"


def test_checkpoint_parity():
    if not os.path.exists(CKPT):
        print("SKIP: checkpoint not present at", CKPT)
        return
    sys.path.insert(0, "/tmp")
    import ligmpnn_model as ref

    from jlig_mpnn.model import LigandMPNN

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=ck["num_edges"],
                        atom_context_num=ck["atom_context_num"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()

    B, L, M = 1, 40, 25
    rng = np.random.RandomState(2)
    arrs = dict(
        X=(rng.randn(B, L, 4, 3) * 5).astype(np.float32), mask=np.ones((B, L), np.float32),
        Y=(rng.randn(B, L, M, 3) * 5).astype(np.float32),
        Y_t=rng.randint(1, 30, (B, L, M)).astype(np.float32),
        Y_m=(rng.rand(B, L, M) > 0.2).astype(np.float32),
        R_idx=np.tile(np.arange(L), (B, 1)).astype(np.float32),
        chain_labels=np.zeros((B, L), np.float32), S=rng.randint(0, 21, (B, L)),
        chain_mask=np.ones((B, L), np.float32), randn=rng.randn(B, L).astype(np.float32))

    fd = {"batch_size": 1, "symmetry_residues": [[]]}
    fd.update({k: torch.tensor(v) for k, v in arrs.items()})
    with torch.no_grad():
        lp_t = m.score(fd, use_sequence=True)["log_probs"].numpy()

    j = LigandMPNN.from_torch(m)
    lp_j = np.asarray(j.score(*[jnp.asarray(arrs[k]) for k in (
        "S", "X", "mask", "Y", "Y_m", "Y_t", "R_idx", "chain_labels", "chain_mask", "randn")]))

    err = np.abs(lp_j - lp_t).max()
    agree = (lp_j.argmax(-1) == lp_t.argmax(-1)).mean()
    assert err < 1e-4 and agree == 1.0, f"err {err} agree {agree}"
    print(f"CHECKPOINT PARITY OK: log_probs max abs err {err:.2e}; argmax agreement {agree:.3f}")


if __name__ == "__main__":
    test_checkpoint_parity()
