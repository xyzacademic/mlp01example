import torch
import torch.nn as nn
import torch.nn.functional as F


def sign(x):
    return x.sign_()
    # return torch.sigmoid(x)


def signb(x):
    return F.relu_(torch.sign(x)).float()
    # return x.float()


def softmax_(x):
    return F.softmax(x.float(), dim=-1)


def sigmoid_(x):
    return torch.sigmoid(x)


class MySign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        # scale = math.sqrt(inputs.size(1)) * 3
        grad_output[inputs.abs() > inputs.abs().mean() / 1] = 0
        return grad_output


msign = MySign.apply


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
            out = out.mean(dim=1)
        else:
            out = self.signb(out)
            out = out.mean(dim=1)

        return out


arch = {}

arch['mlp01scale'] = mlp01scale
