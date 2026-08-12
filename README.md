# jlig_mpnn

JAX/Equinox port of LigandMPNN (Dauparas et al.), loaded from the torch checkpoints via
`from_torch`. Gives a differentiable, ligand-aware sequence log-likelihood usable as an
in-loop loss for gradient-based protein design.

Port of https://github.com/dauparas/LigandMPNN — not affiliated.

## Status
WIP. Porting the `ligand_mpnn` model: encoder + ligand atom-context branch
(`ProteinFeaturesLigand`, context encoder) + decoder, with numerical parity against the
torch reference.

## Layout
- `src/jlig_mpnn/` — the JAX model + `from_torch` loader

Built on jax + equinox. Weights: the LigandMPNN torch checkpoints (`ligandmpnn_v_32_*`).
