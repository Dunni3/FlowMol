This directory contains trained models. Each trained model is packaged in a directory with the following structure:

```bash
model_directory:
    checkpoints/last.ckpt
    config.yaml
```

| Model Directory | Model Description |
|--------------|--------------|
| flowmol3 | FlowMol3 trained on the GEOM-Drugs dataset. |
| fm3_nofa | FlowMol3 trained without fake atoms. |
| fm3_nodistort | FlowMol3 trained without geometry distrotion. |
| fm3_nosc | FlowMol3 trained without self-conditioning. |
| fm3_none | FlowMol3 trained without any of the self-correcting features (geometry distortion, fake atoms, self-conditioning). |


# Download All Pre-Trained Models

Run the following command **from the root of this repository**:

```console
wget -r -np -nH --cut-dirs=3 --reject 'index.html*' -P flowmol/trained_models/ https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3/
```

Trained models are now stored in the  `flowmol/trained_models/` directory. 