<h1 align="center"> Tree Prompting Experiments </h1>
<p align="center"> Tree Prompting: Efficient Task Adaptation without Fine-Tuning, code for the <a href="">Tree-prompt paper</a>. 
</p>

This repo contains code for reproducing experiments in the <a href="">Tree-prompt paper</a>. For a simple, easy-to-use interface, see https://github.com/csinva/tree-prompt.

# Reproducing experiments

### Organization
- `tprompt`: contains main code for modeling (e.g. model architecture)
- `experiments`: code for runnning experiments (e.g. loading data, training models, evaluating models)
- `scripts`: scripts for running experiments (e.g. python scripts that launch jobs in `experiments` folder with different hyperparams)
- `notebooks`: jupyter notebooks for analyzing results and making figures
- `tests`: unit tests

### Setup
- clone and run `pip install -e .`, resulting in a package named `tprompt` that can be imported
    - see `setup.py` for dependencies, not all are required
- example run: run `python scripts/01_train_basic_models.py` (which calls `experiments/01_train_model.py` then view the results in `notebooks/01_model_results.ipynb`
