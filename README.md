# CV24_diffusion-confsolv

This repo contains the code for the work "Torsional Diffusion for Solvated Molecular Conformation Generation". Some of the code is adpated from the original torsional-diffusion repo (https://github.com/gcorso/torsional-diffusion) and the original conf_solv repo (https://github.com/PattanaikL/conf_solv).

Setting up Conda environment
```
conda env create -f environment.yml
conda activate cv24
```

Install e3nn using pip:
```
pip install e3nn
```

# Finetuning torsional diffusion model
```
python train.py \
    --in_node_features=74 \
    --lr=1e-4 \
    --restart_dir=./DRUGS_models
    --data_dir=data/H2O_Pickle \
    --cache=data/H2O_Pickle \
    --log_dir=H2O_models
```

# Evaluating torsional diffusion model

# Training Boltzmann generator with ConfSolv energies
```
python train.py \
    --boltzmann_training \
    --boltzmann_weight \
    --sigma_min 0.1 \
    --temp 500 \
    --log_dir boltz_solv \
    --data_dir data/H2O_Pickle \
    --cache data/H2O_Pickle \
    --restart_dir H2O_models \
    > log.boltz.train 2>&1&
```

# Evaluating Boltzmann generator
```
python test_boltzmann.py \
    --model_dir boltz_solv \
    --temp 500 \
    --model_steps 20 \
    --original_model_dir /workdir/model_solv \
    --test_pkl H2O_models \
    --out boltzmann.out
```


