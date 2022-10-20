"""

Initialization module
"""

import torch


def init_bias(net, data):
    layers = list(net._modules.keys())
    # initialize layer in order

    for layer in layers[:-1]:
        # set bias as projection's mean
        projection = net(data, input_=layers[0], layer=layer + '_projection')
        if 'fc' in layer:
            net._modules[layer].bias = torch.nn.Parameter(
                projection.mean(dim=0), requires_grad=False)
        elif 'conv' in layer:
            net._modules[layer].bias = torch.nn.Parameter(
                projection.transpose_(0, 1).reshape((projection.size(0), -1)).mean(dim=1),
                requires_grad=False)
        del projection
    return None


def init_bias_last_layer(net, data, layer, criterion, target, dtype, input_=None):
    p2 = net(data, input_=input_, layer=layer + '_projection')
    unique_p2 = torch.unique(p2, sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

    new_projection = p2 + temp_bias.reshape((1, -1))
    yp = net(new_projection, input_=layer + '_ap')
    loss_group = criterion(yp, target)
    best_index = loss_group.argmin()
    best_bias = temp_bias[best_index]
    net._modules[layer].bias.data.fill_(best_bias)

    return loss_group[best_index].item(), best_bias
