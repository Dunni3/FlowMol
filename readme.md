# getting the data
we are using the [GEOM-Drugs splits provided by MiDi](https://github.com/cvignac/MiDi#datasets)

# processing a split of the geom dataset

```console
python process_geom.py data/raw/test_data.pickle --config configs/dev.yml
```

# TODO:
- [ ] implement training loop
- [ ] implement molecule sampling
- [ ] implement molecule evaluation
- [ ] make interpolants stochastic + add score-matching loss