# Train for MLP01SCD


## Prepare


## Example

### Single Run
    python train_cnn01_01.py --nrows 0.75 --localit 1 --updated_fc_features 128 \
    --updated_fc_nodes 1 --width 100 --normalize 1 --percentile 1 --fail_count 1 \
    --loss 01loss --act sign --fc_diversity 1 --init normal --no_bias 0 --scale 1 \
    --w-inc1 0.17 --w-inc2 0.17 --version mlp01scale --seed 0 --iters 1000 \
    --dataset cifar10_binary --n_classes 2 --cnn 0 --divmean 0 \
    --target cifar10_binary_01_mlp01scd_0 --updated_fc_ratio 1 --verbose_iter 50 \
    --c0 0 --c1 1 

#### Parameters Explain:

- `--nrows` 7500/10000 (batch_size)
- `--updated_fc_features` feature pool size is 128
- `--updated_fc_nodes randomly` update 1 node in each iteration
- `--normalize` 1 normalize the weight
- `--loss 01 loss`
- `--init weights` initialization followed by normal distribution
- `--no_bias` all layers have bias
- `--w-inc1 --w-inc2` step-size for the first and the second layer both are 0.17
- `--version` architecture version is mlp01scale
- `--seed` random seed is 0
- `--n_classes` 2 classes
- `--cnn` 0 will flatten the vector
- `--divmean` 0 does not normalize the data
- `--target` model checkpoints and logs name
- `--verbose_iter` print acc and loss every iteration.
- `--dataset` dataset is cifar10_binary
- `--c0` the first class is class0
- `--c1` the second class is class1

### Run for 8 votes
1. run the code above in single run for 8 continuous different random seeds(suggest 0~7)
such as in [run_example_for_8_votes_training.sh](run_example_for_8_votes_training.sh)
    You will get 8 checkpoints in directory `checkpoints/pt/`


        cifar10_binary_01_mlp01scd_0.pt  cifar10_binary_01_mlp01scd_2.pt  cifar10_binary_01_mlp01scd_4.pt  cifar10_binary_01_mlp01scd_6.pt
        cifar10_binary_01_mlp01scd_1.pt  cifar10_binary_01_mlp01scd_3.pt  cifar10_binary_01_mlp01scd_5.pt  cifar10_binary_01_mlp01scd_7.pt


2. Combine the models

    Run the command
    
        python combine_vote_mlp.py --dataset cifar10_binary --n_classes 2 \
        --c0 0 --c1 1 --save --target cifar10_binary_01_mlp01scd --votes 8 \
        --no_bias 0 --scale 1 --cnn 0 --act sign --version mlp01scale

    Terminal outputs:

        cifar10_binary_01_mlp01scd_8.pkl saved successfully
        8 votes train accuracy, 0.9005 inference cost 6.97 seconds:
        8 votes test accuracy: 0.856 inference cost 0.03 seconds:
    
    Then you will get `cifar10_binary_01_mlp01scd_8.pkl` in directory `checkpoints`.


## Customize neural network architecture
Please follow the instructions in [jupyter notebook link](https://github.com/zero-one-loss/scd/blob/master/examples/network_architecture_define.ipynb).
If you wanna attack the customized model, you should copy the architecture definition
from `mlp01example/core/ensemble_model` to `mlp01example/attack_demo/core/ensemble_model.py`. 
And then do a modification for the last several lines.

Example of `mlp01scale`

1. Architecture in `mlp01example/core/ensemble_model`

```python
    class mlp01scaleln(nn.Module):
        def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
            super(mlp01scaleln, self).__init__()
            self.votes = votes
            self.num_classes = num_classes
            if act == "sign":
                self.act = torch.sign
            elif act == "signb":
                self.act = signb
            elif act == "sigmoid":
                self.act = torch.sigmoid_
            elif act == "relu":
                self.act = torch.relu_
    
            if softmax:
                if num_classes < 3:
                    raise ValueError("num_classes expect larger than 3, but got {num_classes}")
                self.signb = softmax_
            else:
                self.signb = torch.sigmoid if sigmoid else signb
    
            self.fc1_si = nn.Conv1d(1, 20 * votes, kernel_size=3072, bias=bias)
            self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
            self.layers = ["fc1_si", "fc2_si"]
    
        def forward(self, out):
            out.unsqueeze_(dim=1)
            out = self.fc1_si(out)
    
            out = out.reshape((out.size(0), self.votes, -1))
            out = msign((out - out.mean(dim=2, keepdims=True))/out.std(dim=2, keepdims=True)) * 0.2236
            out = self.fc2_si(out)
            out = out.reshape((out.size(0), self.votes, self.num_classes))
            if self.num_classes == 1:
                out = self.signb(out).squeeze(dim=-1)
                out = out.mean(dim=1).round()
            else:
                out = self.signb(out)
                out = out.mean(dim=1).argmax(dim=-1)
    
            return out
```



2. Architecture in `mlp01example/attack_demo/core/ensemble_model.py`

```python
    class mlp01scale(nn.Module):
        def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
            super(mlp01scale, self).__init__()
            self.votes = votes
            self.num_classes = num_classes
            if act == "sign":
                self.act = torch.sign
            elif act == "signb":
                self.act = signb
            elif act == "sigmoid":
                self.act = torch.sigmoid_
            elif act == "relu":
                self.act = torch.relu_
    
            if softmax:
                if num_classes < 3:
                    raise ValueError("num_classes expect larger than 3, but got {num_classes}")
                self.signb = softmax_
            else:
                self.signb = torch.sigmoid if sigmoid else signb
    
            self.fc1_si = nn.Conv1d(1, 20 * votes, kernel_size=3072, bias=bias)
            self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
            self.layers = ["fc1_si", "fc2_si"]
    
        def forward(self, out):
            out = out.unsqueeze(dim=1)
            out = self.fc1_si(out)
            out = msign(out) * 0.2236
            out = out.reshape((out.size(0), self.votes, -1))
            out = self.fc2_si(out)
            out = out.reshape((out.size(0), self.votes, self.num_classes))
            if self.num_classes == 1:
                out = self.signb(out).squeeze(dim=-1)
                out = out.mean(dim=1)  # <-- modify here
            else:
                out = self.signb(out)
                out = out.mean(dim=1)  # <-- modify here
    
            return out
```
