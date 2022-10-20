import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import os
import math


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
        grad_output[inputs.abs()>inputs.abs().mean()] = 0
        return grad_output


msign = MySign.apply


class ModelWrapper(object):
    def __init__(self, net):
        self.net = net

    def predict(self, x, batch_size=2000):
        with torch.no_grad():
            if type(x) is not torch.Tensor:
                self.net.float()
                x = torch.from_numpy(x).type_as(self.net._modules[list(self.net._modules.keys())[0]].weight)
            if torch.cuda.is_available():
                self.net.cuda()
                x = x.cuda()

            if batch_size:
                n_batch = x.shape[0] // batch_size
                n_rest = x.shape[0] % batch_size
                yp = []
                for i in range(n_batch):
                    yp.append(
                        self.net(x[batch_size * i: batch_size * (i + 1)]))
                if n_rest > 0:
                    yp.append(self.net(x[batch_size * n_batch:]))
                yp = torch.cat(yp, dim=0)
            else:
                yp = self.net(x)

        return yp.cpu().numpy()


    def predict_proba(self, x, batch_size=None, votes=None):
        pass

    def inference(self, x, prob=False, all=False, votes=None):
        pass


class ModelWrapper2(object):
    def __init__(self, structure, votes, path,):
        self.net = {}
        self.votes = votes
        for i in range(votes):
            self.net[i] = structure()
            self.net[i].load_state_dict(torch.load(path[i],
                            map_location=torch.device('cpu')))

    def predict(self, x, batch_size=2000):

        if batch_size:
            n_batch = x.shape[0] // batch_size
            n_rest = x.shape[0] % batch_size
            yp = []
            for i in range(n_batch):
                # print(i)
                yp.append(
                    self.inference(x[batch_size * i: batch_size * (i + 1)]))
            if n_rest > 0:
                yp.append(self.inference(x[batch_size * n_batch:]))
            yp = torch.cat(yp, dim=0)

        else:
            # yp = self.net(x)
            yp = self.inference(x)

        return yp.cpu().numpy()


    def predict_proba(self, x, batch_size=None, votes=None):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).type_as(
                self.net[0]._modules[list(self.net[0]._modules.keys())[0]].weight)
        if torch.cuda.is_available():
            for i in range(self.votes):
                self.net[i].cuda()
            x = x.cuda()

        yp = []
        for i in range(self.votes):
            yp.append(self.net[i](x))
        yp = torch.stack(yp, dim=1)

        return yp.mean(dim=1).cpu()

    def inference(self, x, prob=False, all=False, votes=None):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).type_as(
                self.net[0]._modules[list(self.net[0]._modules.keys())[0]].weight)
        if torch.cuda.is_available():
            for i in range(self.votes):
                self.net[i].cuda()
            x = x.cuda()

        yp = []
        for i in range(self.votes):
            yp.append(self.net[i](x))
        yp = torch.stack(yp, dim=1)

        return yp.mean(dim=1).argmax(dim=-1).cpu()



class Toy3srr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3srr100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3ssr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3ssr100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3sss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3sss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy2ss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy2ss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc3_si = nn.Conv1d(votes, 100 * votes, kernel_size=512, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "fc3_si", "fc4_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 4)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc3_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy2sr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy2sr100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc3_si = nn.Conv1d(votes, 100 * votes, kernel_size=512, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "fc3_si", "fc4_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 4)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc3_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rrs100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rrs100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rrr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rrr100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rsr100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rsr100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rrss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rrss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rsss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rsss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rsrs100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rsrs100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class WordCNN01(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(WordCNN01, self).__init__()
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
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(1, 150 * votes, kernel_size=(4, 200), padding=(2, 0), bias=bias)
        self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=150, bias=bias, groups=votes)
        self.layers = ["conv1_si", "fc2_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out.squeeze_(dim=-1)
        out = F.avg_pool1d(out, out.size(2))
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1)

        return out


class toy3ssss100(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssss100, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3rrs100as(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3rrs100as, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2).sign()
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3rrss100as(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3rrss100as, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.avg_pool2d(out, 2).sign()
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3rrs100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3rrs100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3rrss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3rrss100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out

class Toy3rsr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rsr100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rss100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rsss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rsss100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.sign(out)
        # out = F.avg_pool2d(out, 2).sign()
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3srr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3srr100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssr100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssr100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3sss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3sss100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssss100fs(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssss100fs, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv2_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = self.conv3_si(out)
        out = torch.sign(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 2) * 4
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.sign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3srr100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3srr100scale, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssr100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssr100scale, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3sss100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3sss100scale, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssss100scale(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssss100scale, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = msign(out) * 0.1000
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3srr100linear(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3srr100linear, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = out * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssr100linear(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssr100linear, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = out * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = out * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3sss100linear(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3sss100linear, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = out * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = out * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = out * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3ssss100linear(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3ssss100linear, self).__init__()
        self.votes = votes
        self.num_classes = num_classes
        if act == "sign":
            self.act = msign
        elif act == "signb":
            self.act = signb
        elif act == "sigmoid":
            self.act = torch.sigmoid_
        elif act == "relu":
            self.act = torch.relu_

        if softmax:
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = out * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = out * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = out * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = out * 0.1000
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out

class Toy3rrr100ap2(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rrr100ap2, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = F.avg_pool2d(out, 2)
        out = torch.relu_(out)
        out = self.conv2_si(out)
        out = F.avg_pool2d(out, 2)
        out = torch.relu_(out)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class Toy3rrr100ap1(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(Toy3rrr100ap1, self).__init__()
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
            if num_classes < 2:
                raise ValueError("num_classes expect larger than 3, but got {num_classes}")
            self.signb = softmax_
        else:
            self.signb = torch.sigmoid if sigmoid else signb

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=1024, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = F.avg_pool2d(out, 2)
        out = torch.relu_(out)
        out = torch.relu_(self.conv2_si(out))
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlp01(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlp01, self).__init__()
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
        out = msign(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlpsr(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlpsr, self).__init__()
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
        out = torch.sigmoid(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlprr(nn.Module):
    def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlprr, self).__init__()
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
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out




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
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = msign(out) * 0.2236
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlp2srscale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlp2srscale, self).__init__()
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
        self.fc2_si = nn.Conv1d(votes, 20 * votes, kernel_size=20, bias=bias, groups=votes)
        self.fc3_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = msign(out) * 0.2236
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc3_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlp2ssscale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlp2ssscale, self).__init__()
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
        self.fc2_si = nn.Conv1d(votes, 20 * votes, kernel_size=20, bias=bias, groups=votes)
        self.fc3_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = msign(out) * 0.2236
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = msign(out) * 0.2236
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc3_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlp2rr(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlp2rr, self).__init__()
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
        self.fc2_si = nn.Conv1d(votes, 20 * votes, kernel_size=20, bias=bias, groups=votes)
        self.fc3_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si", "fc3_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc3_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy4rrr100(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy4rrr100, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv4_si = nn.Conv2d(64 * votes, 128 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, 100 * votes, kernel_size=4608, bias=bias, groups=votes)
        self.fc6_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "conv4_si", "fc5_si", "fc6_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv4_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc6_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy4srr100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy4srr100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv4_si = nn.Conv2d(64 * votes, 128 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, 100 * votes, kernel_size=4608, bias=bias, groups=votes)
        self.fc6_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "conv4_si", "fc5_si", "fc6_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv4_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc6_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy4ssr100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy4ssr100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv4_si = nn.Conv2d(64 * votes, 128 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, 100 * votes, kernel_size=4608, bias=bias, groups=votes)
        self.fc6_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "conv4_si", "fc5_si", "fc6_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv4_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc6_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy4sss100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy4sss100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv4_si = nn.Conv2d(64 * votes, 128 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, 100 * votes, kernel_size=4608, bias=bias, groups=votes)
        self.fc6_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "conv4_si", "fc5_si", "fc6_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = self.conv4_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc6_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3gssr100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3gssr100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=2304, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3gsss100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3gsss100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=2304, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3gsrr100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3gsrr100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=2304, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3grrr100(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3grrr100, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=2304, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = torch.relu_(out)
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy4ssss100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy4ssss100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv4_si = nn.Conv2d(64 * votes, 128 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, 100 * votes, kernel_size=4608, bias=bias, groups=votes)
        self.fc6_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "conv4_si", "fc5_si", "fc6_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = self.conv4_si(out)
        out = msign(out) * 0.0295
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc6_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class toy3gssss100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(toy3gssss100scale, self).__init__()
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

        self.conv1_si = nn.Conv2d(3, 16 * votes, kernel_size=3, padding=1, bias=bias)
        self.conv2_si = nn.Conv2d(16 * votes, 32 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.conv3_si = nn.Conv2d(32 * votes, 64 * votes, kernel_size=3, padding=1, bias=bias, groups=votes)
        self.fc4_si = nn.Conv1d(votes, 100 * votes, kernel_size=2304, bias=bias, groups=votes)
        self.fc5_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["conv1_si", "conv2_si", "conv3_si", "fc4_si", "fc5_si"]

    def forward(self, out):
        out = self.conv1_si(out)
        out = msign(out) * 0.0833
        out = F.avg_pool2d(out, 2)
        out = self.conv2_si(out)
        out = msign(out) * 0.0589
        out = F.avg_pool2d(out, 2)
        out = self.conv3_si(out)
        out = msign(out) * 0.0417
        out = F.avg_pool2d(out, 2)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc4_si(out)
        out = msign(out) * 0.1000
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc5_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlpcs20scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlpcs20scale, self).__init__()
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

        self.fc1_si = nn.Conv1d(1, 20 * votes, kernel_size=27648, bias=bias)
        self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = msign(out) * 0.2236
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlpcr20scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlpcr20scale, self).__init__()
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

        self.fc1_si = nn.Conv1d(1, 20 * votes, kernel_size=27648, bias=bias)
        self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=20, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlpcr100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlpcr100scale, self).__init__()
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

        self.fc1_si = nn.Conv1d(1, 100 * votes, kernel_size=27648, bias=bias)
        self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = torch.relu_(out)
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


class mlpcs100scale(nn.Module):
    def __init__(self, num_classes=2, act="sign", sigmoid=False, softmax=False, scale=1, votes=1, bias=True):
        super(mlpcs100scale, self).__init__()
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

        self.fc1_si = nn.Conv1d(1, 100 * votes, kernel_size=27648, bias=bias)
        self.fc2_si = nn.Conv1d(votes, num_classes * votes, kernel_size=100, bias=bias, groups=votes)
        self.layers = ["fc1_si", "fc2_si"]

    def forward(self, out):
        out.unsqueeze_(dim=1)
        out = self.fc1_si(out)
        out = msign(out) * 0.1000
        out = out.reshape((out.size(0), self.votes, -1))
        out = self.fc2_si(out)
        out = out.reshape((out.size(0), self.votes, self.num_classes))
        if self.num_classes == 1:
            out = self.signb(out).squeeze(dim=-1)
            out = out.mean(dim=1).round()
        else:
            out = self.signb(out)
            out = out.mean(dim=1).argmax(dim=-1)

        return out


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


arch = {}


arch['toy3srr100'] = Toy3srr100
arch['toy3ssr100'] = Toy3ssr100
arch['toy3sss100'] = Toy3sss100
arch['toy2ss100'] = Toy2ss100
arch['toy2sr100'] = Toy2sr100

arch['toy3rrs100'] = Toy3rrs100
arch['toy3rrr100'] = Toy3rrr100
arch['toy3rsr100'] = Toy3rsr100
arch['toy3rss100'] = Toy3rss100
arch['toy3rrss100'] = Toy3rrss100
arch['toy3rsss100'] = Toy3rsss100
arch['toy3rsrs100'] = Toy3rsrs100
arch['wordcnn01'] = WordCNN01
arch['toy3ssss100'] = toy3ssss100

arch['toy3rrs100as'] = toy3rrs100as
arch['toy3rrss100as'] = toy3rrss100as
arch['toy3rrs100fs'] = toy3rrs100fs
arch['toy3rrss100fs'] = toy3rrss100fs

arch['toy3rsr100fs'] = Toy3rsr100fs
arch['toy3rss100fs'] = Toy3rss100fs
arch['toy3rsss100fs'] = Toy3rsss100fs

arch['toy3srr100fs'] = toy3srr100fs
arch['toy3ssr100fs'] = toy3ssr100fs
arch['toy3sss100fs'] = toy3sss100fs
arch['toy3ssss100fs'] = toy3ssss100fs

arch['toy3srr100scale'] = toy3srr100scale
arch['toy3ssr100scale'] = toy3ssr100scale
arch['toy3sss100scale'] = toy3sss100scale
arch['toy3ssss100scale'] = toy3ssss100scale
arch['toy3srr100linear'] = toy3srr100linear
arch['toy3ssr100linear'] = toy3ssr100linear
arch['toy3sss100linear'] = toy3sss100linear
arch['toy3ssss100linear'] = toy3ssss100linear
arch['toy3rrr100ap1'] = Toy3rrr100ap1
arch['toy3rrr100ap2'] = Toy3rrr100ap2
arch['mlp01'] = mlp01
arch['mlpsr'] = mlpsr
arch['mlprr'] = mlprr

arch['mlp01scale'] = mlp01scale
arch['mlp01scaleln'] = mlp01scaleln
arch['mlp2srscale'] = mlp2srscale
arch['mlp2ssscale'] = mlp2ssscale


arch['toy4rrr100'] = toy4rrr100
arch['toy4srr100scale'] = toy4srr100scale
arch['toy4ssr100scale'] = toy4ssr100scale
arch['toy4sss100scale'] = toy4sss100scale
arch['toy3gssr100scale'] = toy3gssr100scale
arch['toy3gsss100scale'] = toy3gsss100scale
arch['toy3gsrr100scale'] = toy3gsrr100scale
arch['toy3grrr100'] = toy3grrr100
arch['toy4ssss100scale'] = toy4ssss100scale
arch['toy3gssss100scale'] = toy3gssss100scale
arch['mlpcs20scale'] = mlpcs20scale
arch['mlpcr20scale'] = mlpcr20scale
arch['mlpcr100scale'] = mlpcr100scale
arch['mlpcs100scale'] = mlpcs100scale

if __name__ == '__main__':
    x = torch.rand(size=(1000, 3, 32, 32))
    # x = torch.rand(size=(1000, 3072))

