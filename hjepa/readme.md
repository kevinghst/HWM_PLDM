## Two rooms

The following command runs JEPA training on dataset_size=3M, sequence_length=17. Full list of configs can be found in `configs/wall/icml/`. Change the `config` field to replicate other experiments from the paper.

```
python train.py --config configs/wall/icml/seqlen90_3M.yaml
```


Yaml files override each other if they share values, with the last element in the list overriding last.
Values list allows to modify default or loaded configs. To change the learning rate and batch size for `fixed_wall.yaml`, you can do:
```
python train.py --config configs/wall/icml/seqlen90_3M.yaml --values base_lr=0.01 data.offline_wall_config.batch_size=128
```

## Diverse Mazes

The following command runs JEPA training on the 5 maps setting. Full list of configs can be found in `configs/diverse_maze/icml/`. 

```
python train.py --config configs/diverse_maze/icml/small_diverse_5maps.yaml
```

## Hyperparameter tuning

Hyperparameters ($\alpha, \beta, \lambda, \delta, \omega$) should be tuned for any new environment.

Within a given environment, hyperparameters should be tuned for different offline datasets that have significant differences in data distributions.

To reduce the hyperparamter search space for a given setting, one idea is to take the hyperparameters for the closest setting, get lower and upper bounds for each parameter by dividing and multiplying it by a factor (eg: 3) respectively, and perform a random search within the lower and upper bounds of all parameters.
