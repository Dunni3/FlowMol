# getting the data
we are using the [GEOM-Drugs splits provided by MiDi](https://github.com/cvignac/MiDi#datasets)

# processing a split of the geom dataset

```console
python process_geom.py data/raw/test_data.pickle --config configs/dev.yml
```

# TODO:
- [ ] implement model saving via pytorch lightning - current checkpointing is done every n epochs - want something better probably
- [ ] is there an automatic way to compute loss on a test set?
- [ ] put model sampling / eval into training loop
- [ ] do interpolation weights become unstable at the end of integration?
- [ ] double check derivative of interpolation weights
- [ ] implement molecule evaluation (frag frac, valid atoms, midi valency calculations)
- [ ] make interpolants stochastic + add score-matching loss
- [ ] does epoch_exact get aligned with the vaidaiton loss parameters even if i don't log epoch_exact during validaiton steps? (i think so, need to see if plots come out ok on wandb)