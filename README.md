# jlig_mpnn

JAX/Equinox port of LigandMPNN (Dauparas et al.), loaded from the released torch checkpoints
via `from_torch`. Gives a **differentiable, ligand-aware** sequence log-likelihood — usable as
an in-loop loss for gradient-based protein design, which the torch model is not.

Port of https://github.com/dauparas/LigandMPNN not affiliated.

## Use
```python
model = LigandMPNN.from_torch(torch_ligand_mpnn)   # a loaded ligandmpnn_v_32_* checkpoint
log_probs = model.score(S, X, mask, Y, Y_m, Y_t, R_idx, chain_labels, chain_mask, randn)
log_probs = model.score_soft(soft20, ...)          # differentiable in the sequence
```
`score` teacher-forces an integer sequence; `score_soft` takes a soft `[..,20]` sequence
(native LigandMPNN residue order) so gradients flow to it. Both condition on the backbone and
the ligand atom context. Scope: `model_type="ligand_mpnn"`, `use_side_chains=False`.

## Parity
Verified against the torch reference on CPU, block by block:
- featurizer (kNN graph, RBFs, ligand context): `E_idx` exact, tensors < 1e-4
- full `score` log-probs vs the real `ligandmpnn_v_32_010_25` checkpoint: < 3e-5, argmax identical
- `score_soft(one_hot(S)) == score(S)` exactly; gradient wrt the sequence is finite and nonzero

`python tests/test_*_parity.py` (needs torch + the LigandMPNN reference module + checkpoint).

## Layout
- `src/jlig_mpnn/` — `model`, `features`, `layers`, `backend` (the `from_torch` machinery)
- `tests/` — parity vs the torch reference

Deps: jax + equinox.
