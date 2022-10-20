from dataclasses import dataclass, field, fields, asdict
from multiprocessing import cpu_count
import json
import sys
import os
import torch

@dataclass
class ModelArgs:

    target: str = 'models.pkl'
    nrows: float = 0.75
    nfeatures: float = 1.0
    w_inc: float = 0.17
    tol: float = 0.00000
    local_iter: int = 100
    num_iters: int = 1000
    interval: int = 20
    rounds: int = 100
    w_inc1: float = 0.17
    updated_features: int = 128
    updated_fc_features: int = 128
    updated_conv_features: int = 8
    n_jobs: int = 1
    num_gpus: int = 1
    adv_train: bool = False
    eps: float = 0.1
    w_inc2: float = 0.2
    hidden_nodes: int = 20
    evaluation: bool = True
    verbose: bool = True
    b_ratio: float = 0.2
    cuda: bool = True
    seed: int = 2018
    save: bool = False
    criterion: classmethod = torch.nn.Module
    structure: classmethod = torch.nn.Module
    dataset: str = 'mnist'
    num_classes: int = 2
    c: float = 1.0
    gpu: int = 0
    fp16: bool = False
    resume: bool = False
    act: str = 'sign'
    verbose: bool = False
    lr: float = 0.01
    percentile: bool = False
    sigmoid: bool = False
    updated_fc_ratio: int = 1
    updated_conv_ratio: int = 1
    updated_fc_nodes: int = 1
    updated_conv_nodes: int = 1
    softmax: bool = False
    fail_count: int = 1
    width: int = 100
    normalize: bool = False
    loss: str = '01'
    diversity: bool = False
    conv_diversity: bool = False
    fc_diversity: bool = False
    updated_conv_features_diversity: int = 1
    diversity_train_stop_iters: int = 3000
    init: str = 'normal'
    scale: int = 1
    save_path: str = 'pt'
    adaptive_loss_epoch: int = 0
    inc_version: str = 'v1'
    mean_only: int = 0
    divmean: int = 0
    freeze_layer: int = 0
    temp_save_per_iter: int = 0
    lr_decay_iter: int = 0
    batch_increase_iter: int = 0
    batch_size: int = 256
    aug: int = 0
    balanced_sampling: int = 1
    embedding_path: str = 'glove.6B.200d.txt'
    drop_p: float = 0
    bnn_layer: int = 0
    epsilon: float = 0.0
    alpha: float = 0.0
    step: int = 10
    bp_layer: int = 0
    reinit: int = 0
    bnn_layers: list = field(default_factory=list)
    adv_source: str = ''
    adv_structure: str = ''
    pnorm: str = 'l2'


    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            json.dump(asdict(self), f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)

