# jligandmpnn

[![CI](https://github.com/ssiddhantsharma/jligandmpnn/actions/workflows/ci.yml/badge.svg)](https://github.com/ssiddhantsharma/jligandmpnn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

JAX/Equinox port of [LigandMPNN](https://github.com/dauparas/LigandMPNN) (Dauparas et al.),
the atomic-context protein sequence design model. Loads the released torch checkpoints via
`from_torch` and adds a **differentiable soft-sequence** path, so the ligand-aware sequence
log-likelihood is usable as an in-loop loss for gradient-based design (the torch model is not
differentiable through the sequence). Not affiliated with the authors.

## Install
```bash
pip install -e .        # jax + equinox
```
Weight conversion needs the PyTorch LigandMPNN reference, vendored under `reference/`; it is
imported only to build the torch model and copy its weights. `torch` is the CPU build.

## Usage
Runnable end-to-end example: [`examples/score.py`](examples/score.py).

```python
from jligandmpnn.model import LigandMPNN

model = LigandMPNN.from_torch(torch_ligand_mpnn)   # a loaded ligandmpnn_v_32_* checkpoint
log_probs = model.score(S, X, mask, Y, Y_m, Y_t, R_idx, chain_labels, chain_mask, randn)
log_probs = model.score_soft(soft20, ...)          # differentiable in the sequence
```
`score` teacher-forces an integer sequence; `score_soft` takes a soft `[..,20]` sequence in
native LigandMPNN residue order, so gradients flow to it. Both condition on the backbone and
the ligand atom context. Scope: `model_type="ligand_mpnn"`, `use_side_chains=False`.

## Parity
Verified against the torch reference, block by block (run in CI):
- featurizer (kNN graph, RBFs, ligand context): `E_idx` exact, tensors < 1e-4
- `score` log-probs vs the real `ligandmpnn_v_32_010_25` checkpoint: < 3e-5, argmax identical
- `score_soft(one_hot(S)) == score(S)` exactly; gradient wrt the sequence is finite and nonzero

![parity](figures/parity.png)

`pytest tests` reproduces it; `python scripts/parity_figure.py` regenerates the plot.

## Layout
- `src/jligandmpnn/` `model`, `features`, `layers`, `backend` (the `from_torch` machinery)
- `reference/` vendored PyTorch LigandMPNN (weight source; see `NOTICE`)
- `tests/`, `examples/`, `scripts/`
