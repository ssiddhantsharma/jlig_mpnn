"""End-to-end: load a LigandMPNN checkpoint into JAX and score a sequence.

Weight conversion needs the PyTorch LigandMPNN reference; point LIGMPNN_MODEL_DIR at a
checkout (the dir with its model module importable as `ligmpnn_model`).

  LIGMPNN_MODEL_DIR=/path/to/LigandMPNN \
    python examples/score.py --checkpoint ligandmpnn_v_32_010_25.pt
"""

import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.environ["LIGMPNN_MODEL_DIR"])
import ligmpnn_model as ref

from jligandmpnn.model import LigandMPNN

AA = "ACDEFGHIKLMNPQRSTVWY"  # LigandMPNN residue order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    m = ref.ProteinMPNN(model_type="ligand_mpnn", k_neighbors=ck["num_edges"],
                        atom_context_num=ck["atom_context_num"])
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    model = LigandMPNN.from_torch(m)  # torch -> JAX, once

    # a small synthetic complex: L residues (N,CA,C,O backbone) + M ligand atoms
    L, M = 20, ck["atom_context_num"]
    rng = np.random.RandomState(0)
    args_ = {
        "S": jnp.asarray(rng.randint(0, 20, (1, L))),
        "X": jnp.asarray((rng.randn(1, L, 4, 3) * 5).astype(np.float32)),
        "mask": jnp.ones((1, L)),
        "Y": jnp.asarray((rng.randn(1, L, M, 3) * 5).astype(np.float32)),
        "Y_m": jnp.asarray((rng.rand(1, L, M) > 0.2).astype(np.float32)),
        "Y_t": jnp.asarray(rng.randint(1, 30, (1, L, M)).astype(np.float32)),
        "R_idx": jnp.tile(jnp.arange(L), (1, 1)).astype(jnp.float32),
        "chain_labels": jnp.zeros((1, L)),
        "chain_mask": jnp.ones((1, L)),
        "randn": jnp.asarray(rng.randn(1, L).astype(np.float32))}

    log_probs = model.score(args_["S"], args_["X"], args_["mask"], args_["Y"], args_["Y_m"],
                            args_["Y_t"], args_["R_idx"], args_["chain_labels"],
                            args_["chain_mask"], args_["randn"])[0]  # [L, 21]
    argmax = "".join(AA[i] for i in np.asarray(log_probs[:, :20]).argmax(-1))
    print("log_probs:", log_probs.shape)
    print("LigandMPNN-preferred sequence for this backbone+ligand:", argmax)


if __name__ == "__main__":
    main()
