# getting the data
we are using the [GEOM-Drugs splits provided by MiDi](https://github.com/cvignac/MiDi#datasets)

# processing a split of the geom dataset

```console
python process_geom.py data/raw/test_data.pickle --config configs/dev.yml
```

# TODO:
- [ ] do interpolation weights become unstable at the end of integration?
- [ ] double check derivative of interpolation weights
- [ ] make interpolants stochastic + add score-matching loss
- [ ] play with time loss weights
- [ ] play with prior position distribution variance
- [ ] implement OT alignment
  

## multi-gpu training and where to put the periodic sampling
- periodic sampling is done inside of the training_step call. this means that if we have multiple gpus, we will be doing the sampling multiple times. this is a bottleneck.
- one option is to do the periodic sampling inside the validation_step, possibly. hopefully this we avoid the multi-gpu issue mentioned in the previous bullet point.
- however, when we decide to do sampling is dependent on the epoch/batch that we are at in the trainig loop. Inside the validaiton step, pytorch-lightning does not provide us access to the current/last batch of the training set.
- we could manually record this be setting a variable in the training loop, but this may also create issues with multi-gpu training?? tbd, a problem for future me