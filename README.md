# EMO
This repository contains the source code of our paper 'EMO: Epigraph-Based Multilevel Optimization for Enhancing Chain of Thought Inference Capabilities'. Part of the code is borrowed from [Dissecting-CoT](https://github.com/yingcong-li/Dissecting-CoT).


## Start up
- Set up the environments by running
    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

- Replace wandb user name in config files.

## Training
1. Enter into directory ```src```
2. For baselines `penalty` and `vanilla` 
    Run ```train.py``` based on the choosen config file: ```python train.py --config conf/[config_file].yaml```
   
   For `epigraph`, 
   Run ```train_epigraph.py``` based on the choosen config file: ```python train_epigraph.py --config conf/[config_file].yaml```
python train.py --config conf/cot_2nn.yaml
## Test
1. Enter into directory ```src```
2. Run ```evaluate.py``` after entering the run_id of the model to be evaluated