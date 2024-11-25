This directory contains trained models. Each trained model is packaged in a directory with the following structure:

```bash
model_directory:
    checkpoints/model.ckpt
    config.yaml
```

| Model Directory | Model Description |
|--------------|--------------|
| geom_ctmc | Trained on the GEOM-Drugs dataset. CTMC flows for categorical features. |
| geom_gaussian | Trained on the GEOM-Drugs dataset. Gaussian categorical priors. The "continuous" method from MLSB paper. |
| geom_simplexflow | Trained on the GEOM-Drugs dataset. SimplexFlow using marginal-simplex prior for categorical features. |
| geom_dirichlet | Trained on the GEOM-Drugs dataset. Uses Dirichlet Flow Matching for categorical features. |
| qm9_ctmc | Trained on QM9 dataset. CTMC flows for categorical features. |
| qm9_gaussian | Trained on QM9 dataset. Gaussian categorical priors. The "Continuous" method from MLSB paper. |
| qm9_simplexflow | Trained on QM9 dataset. SimplexFlow using marginal-simplex prior for categorical features. |
| qm9_dirichlet | Trained on QM9 dataset. Uses Dirichlet Flow Matching for categorical features. |