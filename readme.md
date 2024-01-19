# getting the data
we are using the [GEOM-Drugs splits provided by MiDi](https://github.com/cvignac/MiDi#datasets)

# processing a split of the geom dataset

```console
python process_geom.py data/raw/test_data.pickle --config configs/dev.yml
```

# TODO:
- [ ] implement model saving via pytorch lightning
- [ ] do interpolation weights become unstable at the end of integration?
- [ ] double check derivative of interpolation weights
- [ ] implement molecule evaluation (frag frac, valid atoms, midi valency calculations)
- [ ] make interpolants stochastic + add score-matching loss