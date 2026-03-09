## Diverse Mazes

First, download the pretrained level-1 world model weights by running 
```
python scripts/download_ckpt_from_hf.py --out-dir <repo_root>/pretrained
```

To train the level-2 world model on the large-maze setting, run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml
```

Evaluation runs automatically at the end of the training script. If you want to evaluate a trained model later, run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps_l2.yaml --values eval_only=true eval_cfg.probing.load_prober=true load_l1_only=false load_checkpoint_path={output_root}/{output_dir}/___.ckpt
```

If you prefer to train the level-1 world model from scratch, run:

```
python train.py --config configs/diverse_maze/icml/large_diverse_25maps.yaml
```

Then update `load_checkpoint_path` in `configs/diverse_maze/icml/large_diverse_25maps_l2.yaml` to point to your trained level-1 checkpoint.

After that, run the level-2 training command above.

## Hyperparameter tuning

Hyperparameters ($\alpha, \beta, \lambda, \delta, \omega$) should be tuned for any new environment.

Within a given environment, hyperparameters should be tuned for different offline datasets that have significant differences in data distributions.

To reduce the hyperparamter search space for a given setting, one idea is to take the hyperparameters for the closest setting, get lower and upper bounds for each parameter by dividing and multiplying it by a factor (eg: 3) respectively, and perform a random search within the lower and upper bounds of all parameters.
