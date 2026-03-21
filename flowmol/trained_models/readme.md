This directory contains trained models. Each trained model is packaged in a directory with the following structure:

```bash
model_directory:
    checkpoints/last.ckpt
    config.yaml
```

| Model Directory | Model Description |
|---|---|
| `flowmol3` | FlowMol3 default model trained on GEOM-Drugs (geometry distortion: p=0.7, t=0.25). |
| `fm3_nodistort` | FlowMol3 without geometry distortion. |
| `fm3_none` | FlowMol3 without any self-correcting features (no geometry distortion, no fake atoms, no self-conditioning). |
| `fm3_ahigh` | Ablation: high loss weight for atom type (a) feature. |
| `fm3_alow` | Ablation: low loss weight for atom type (a) feature. |
| `fm3_chigh` | Ablation: high loss weight for formal charge (c) feature. |
| `fm3_clow` | Ablation: low loss weight for formal charge (c) feature. |
| `fm3_distort_extreme` | Ablation: extreme geometry distortion (p=1.0, t=0.0). |
| `fm3_distort_highp` | Ablation: high geometry distortion probability (p=0.7, t=0.5). |
| `fm3_distort_hight` | Ablation: high geometry distortion time (p=0.2, t=0.75). |
| `fm3_distort_lowp` | Ablation: low geometry distortion probability (p=0.1, t=0.5). |
| `fm3_distort_lowt` | Ablation: low geometry distortion time (p=0.2, t=0.25). |
| `fm3_ehigh` | Ablation: high loss weight for bond order (e) feature. |
| `fm3_elow` | Ablation: low loss weight for bond order (e) feature. |
| `fm3_fa_highp` | Ablation: high fake atom probability. |
| `fm3_fa_highstd` | Ablation: high fake atom noise standard deviation. |
| `fm3_fa_lowp` | Ablation: low fake atom probability. |
| `fm3_fa_lowstd` | Ablation: low fake atom noise standard deviation. |
| `fm3_scprop_high` | Ablation: high self-conditioning proportion. |
| `fm3_scprop_low` | Ablation: low self-conditioning proportion. |
| `fm3_xhigh` | Ablation: high loss weight for atomic position (x) feature. |
| `fm3_xlow` | Ablation: low loss weight for atomic position (x) feature. |


# Download All Pre-Trained Models

Run the following command **from the root of this repository**:

```console
wget -r -np -nH --cut-dirs=3 --reject 'index.html*' -P flowmol/trained_models/ https://bits.csb.pitt.edu/files/FlowMol/trained_models_v31/
```

Trained models are now stored in the  `flowmol/trained_models/` directory.
