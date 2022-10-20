import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def sign(x):
    return x.sign()
    # return torch.sigmoid(x)


def signb(x):
    return F.relu(torch.sign(x)).float()
    # return x.float()


def softmax_(x):
    return F.softmax(x.float(), dim=1)


# def softmax_(x):
#     return x.float()


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
        grad_output[inputs.abs() > inputs.abs().mean()] = 0
        return grad_output


msign = MySign.apply


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0, std=1)
        # nn.init.uniform_(m.weight, -1, 1)
        m.weight = torch.nn.Parameter(
            m.weight / torch.norm(
                m.weight.view((m.weight.size(0), -1)),
                dim=1).view((-1, 1, 1, 1))
        )
        # m.weight.requires_grad = False
        if m.bias is not None:
            init.constant_(m.bias, 0)
            # m.bias.requires_grad = False
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def _01_init(model):
    for name, m in model.named_modules():
        if 'si' not in name:
            if isinstance(m, nn.Conv2d):
                m.weight = torch.nn.Parameter(
                    m.weight.sign()
                )
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight = torch.nn.Parameter(
                    m.weight.sign())
                if m.bias is not None:
                    m.bias.data.zero_()


def init_weights(model, kind='normal'):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(
                    m.weight.view((m.weight.size(0), -1)),
                    dim=1).view((-1, 1, 1, 1))
            )
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False

        if isinstance(m, nn.Linear):
            if kind == 'normal':
                nn.init.normal_(m.weight, mean=0, std=1)
            elif kind == 'uniform':
                nn.init.uniform_(m.weight, -1, 1)
            m.weight = torch.nn.Parameter(
                m.weight / torch.norm(m.weight, dim=1, keepdim=True))
            # m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.data.zero_()
                # m.bias.requires_grad = False


class mlp01scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):
        super(mlp01scale, self).__init__()
        if act == "sign":
            self.act = torch.sign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 1, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.fc1_si = nn.Linear(3072, 20, bias=bias)
        self.fc2_si = nn.Linear(20, num_classes, bias=bias)
        self.layers = ["fc1_si", "fc2_si"]
        self.apply(_weights_init)

    def forward(self, x, input_=None, layer=None):
        # check input start from which layer
        status = -1
        for items in self.layers:
            status += 1
            if input_ is None or items in input_:
                break

        # layer 1
        if status < 1:
            if input_ != self.layers[0] + "_ap":
                out = self.fc1_si(x)
            if layer == self.layers[0] + "_projection":
                return out
            if input_ == self.layers[0] + "_ap":
                out = x
            out = msign(out) * 0.2236
            if layer == self.layers[0] + "_output":
                return out

        # layer 2
        if input_ == self.layers[1]:
            out = x
        if status < 2:
            if input_ != self.layers[1] + "_ap":
                out = self.fc2_si(out)
            if layer == self.layers[1] + "_projection":
                return out
            if input_ == self.layers[1] + "_ap":
                out = x
            out = self.signb(out)

        return out


arch = {}
arch['mlp01scale'] = mlp01scale
