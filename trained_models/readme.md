This directory contains trained models. Each trained model is packaged in a directory with the following structure:

```bash
model_directory:
    checkpoints/model.ckpt
    config.yaml
```

| Model Directory | Model Description |
|--------------|--------------|
| qm9_gaussian | Trained on QM9 dataset. Gaussian categorical priors. Used for results in Table 1 and Table 2 of the paper. |
| qm9_marginal | Trained on QM9 dataset. Marginal-simplex categorical priors. Used for results in Table 1 of the paper. |
| qm9_dirichlet | Trained on QM9 dataset. Uses Dirichlet Flow Matching for categorical features. Used for results in Table 1 of the paper. |
| geom_gaussian | Train on the GEOM-Drugs dataset. Gaussian categorical priors. Used for results in Table 2 and Table 3 of the paper. |