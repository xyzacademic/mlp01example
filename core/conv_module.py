import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_diversity_index(data, layer, net, target):
    out = net(data, input_=layer, layer=layer+'_projection')
    k = out.reshape((out.size(0), out.size(1), -1)).sign()
    k0 = k[target == 0]
    k1 = k[target == 1]
    t = (k0 == k1).float().mean(dim=-1)
    p = t.mean(dim=0)
    # print(f'{layer} {p.argmax().item()}\'s nodes updating')
    # print('Diversity: ', p)
    # print('\n')
    return p.argmax()


def init_conv(net, data, layer, criterion, target, dtype, idx, scd_args):
    p2 = net(data, input_=layer, layer=layer + '_projection')
    n_nodes = p2.size(1)
    height = p2.size(2)
    width = p2.size(3)
    if not scd_args.percentile:
        unique_p2 = torch.unique(p2[:, idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

        if temp_bias.size(0) < scd_args.width:
            subset_bias = temp_bias
        else:
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            subset_bias = torch.from_numpy(
                np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
    else:
        sorted_pti = p2[:, idx].flatten().sort()[0]
        percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
        unique_p2 = torch.from_numpy(
            np.quantile(sorted_pti.cpu(), percentile)).type_as(p2)
        subset_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        del unique_p2

    new_projection = p2[:, idx:idx + 1] + subset_bias.reshape((1, -1, 1, 1))
    # new_projection shape nrows(1500) * n_bias(100) * h(32) * w(32)
    new_p2 = torch.repeat_interleave(
        p2.unsqueeze_(dim=1), subset_bias.size(0), dim=1)
    del p2
    # replace original idx's projection by variations with different bias
    new_p2[:, :, idx] = new_projection
    del new_projection
    nrows = new_p2.size(0)
    n_variations = new_p2.size(1)
    # switch dimension of nrows and n_bias, then flatten these two dimension

    # n_bias * nrows * nodes * H * W
    new_projection = new_p2.transpose_(0, 1).reshape(
        (-1, n_nodes, height, width))
    del new_p2
    # get the final output, and reverse to original dimension order  nrows * n_bias * 1
    yp = net(new_projection, input_=layer + '_ap').reshape((n_variations, nrows, -1)).transpose(0,
                                                                                                1)
    del new_projection
    loss_group = criterion(yp, target)
    best_index = loss_group.argmin()
    best_bias = subset_bias[best_index]
    try:
        net._modules[layer].bias[idx].fill_(best_bias)
    except:
        pass
    # print(subset_bias)
    # scd_args.logs.append(loss_group.cpu().numpy())
    # scd_args.bias_logs.append(subset_bias.cpu().numpy())
    # print(f'{layer} {idx} nodes: {best_bias}')
    if scd_args.record:
        loss_name = f'{layer}_loss'
        bias_name = f'{layer}_bias'
        if loss_name not in scd_args.logs:
            scd_args.logs[loss_name] = []
        if bias_name not in scd_args.logs:
            scd_args.logs[bias_name] = []
        scd_args.logs[loss_name].append(loss_group.cpu().numpy())
        scd_args.logs[bias_name].append(subset_bias.cpu().numpy())
    del subset_bias
    return loss_group[best_index], best_bias


def update_mid_layer_conv(net, layers, layer_index, data_, dtype, scd_args,
                          criterion, target, device):
    layer = layers[layer_index]
    if layer_index == len(layers) - 1:
        data = data_
    else:
        previous_layer = layers[layer_index + 1]
        data = net(data_, input_=layers[-1], layer=previous_layer + '_output')

    num_nodes = net._modules[layer].weight.shape[0]
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:min(num_nodes, scd_args.updated_conv_nodes)]:
        if scd_args.conv_diversity:
            if scd_args.diversity:
                idx = get_diversity_index(data, layer, net, target)
            train_match_rate = update_conv_diversity(net, layer, layers, layer_index, data, dtype,
                                                 scd_args,
                                                 criterion, target, device, idx)
            print(f'{layer} {idx} nodes match rate: {train_match_rate}')
            # global_bias = net._modules[layer].bias[idx]
            # train_loss = criterion(net(data, input_=layer), target)
        # else:
        try:
            net._modules[layer].bias[idx].zero_()
        except:
            pass
        # Get the global bias for this batch
        train_loss, global_bias = init_conv(net, data, layer, criterion,
                                            target, dtype, idx, scd_args)

        weights = net._modules[layer].weight
        weight_size = weights.size()[1:]
        n_nodes = weight_size[0] * weight_size[1] * weight_size[2]
        best_w = weights[idx:idx + 1].clone()
        w_incs1 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc1
        local_iters = 0
        fail_counts = 0
        while local_iters < scd_args.local_iter and \
                fail_counts < scd_args.fail_count:

            updated_features = min(n_nodes, scd_args.updated_conv_features)
            cords_index = np.random.choice(n_nodes, updated_features, False)
            cords = []
            for i in range(weight_size[0]):
                for j in range(weight_size[1]):
                    for k in range(weight_size[2]):
                        cords.append([i, j, k])
            cords = torch.tensor(cords)[cords_index]

            if 'si' in layer or layer_index == 0 or 'sign' not in scd_args.act:
                inc = []
                for i in range(w_incs1.shape[0]):
                    w_inc = w_incs1[i]
                    w_ = torch.repeat_interleave(
                        best_w, updated_features, dim=0)
                    for j in range(updated_features):
                        w_[i, cords[j][0], cords[j][1], cords[j][2]] += w_inc
                    inc.append(w_)
                w_ = torch.cat(inc, dim=0)
                del inc
                if scd_args.normalize:
                    w_ /= w_.view((updated_features * w_incs1.shape[0], -1)).norm(dim=1).view(
                        (-1, 1, 1, 1))
                # w_ = torch.cat([w_, -1.0 * w_], dim=1)
            else:
                w_incs2 = -1

                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                for i in range(updated_features):
                    w_[i, cords[i][0], cords[i][1], cords[i][2]] *= w_incs2

            ic = w_.shape[0]

            temp_module = torch.nn.Conv2d(in_channels=data.size(1), out_channels=ic, stride=net._modules[layer].stride,
                                          kernel_size=list(weights.size()[2:]),
                                          padding=net._modules[layer].padding).to(dtype=dtype,
                                                                                  device=device)
            temp_module.weight = nn.Parameter(w_)
            temp_module.bias.zero_()
            temp_module.requires_grad_(False)

            # projection's shape  nrows(1500) * ic(96) * H * W

            # del temp_module
            w_batch_size = ic // scd_args.updated_conv_ratio
            loss_group = []
            bias = []
            for j in range(scd_args.updated_conv_ratio):
                projection = temp_module(data)[:, w_batch_size * j:w_batch_size * (j + 1)]
                new_projection, bias_batch = update_conv_weight(
                    projection, global_bias,
                    scd_args)
                bias.append(bias_batch)
                n_r = new_projection.size(0)  # 1500
                n_w = new_projection.size(1)  # 16
                n_b = new_projection.size(2)  # 20
                height = new_projection.size(3)  # 32
                width = new_projection.size(4)
                new_projection = new_projection.reshape((n_r, n_w * n_b, height, width))
                # new_projection 1500*16*20  bias 16*20
                # original projection feed into next layer
                projection = net(data, input_=layer, layer=layer + '_projection')

                # replace projection[:, idx] after flatten variations
                projection = torch.repeat_interleave(projection.unsqueeze_(dim=1), n_w * n_b, dim=1)
                projection[:, :, idx] = new_projection
                del new_projection
                projection = projection.transpose_(0, 1).reshape(
                    (-1, projection.size(2), height, width))
                yp = net(projection, input_=layer + '_ap').reshape((n_w * n_b, n_r, -1))
                del projection
                yp = yp.transpose_(0, 1).reshape((n_r, n_w, n_b, -1))
                loss_group.append(criterion(yp, target))

            bias = torch.cat(bias, dim=0)
            loss_group = torch.cat(loss_group, dim=0)
            # print(loss_group.shape)
            loss_group = loss_group.cpu().numpy()
            new_loss = loss_group.min()
            if new_loss <= train_loss:
                row, col = np.unravel_index(loss_group.argmin(), loss_group.shape)
                net._modules[layer].weight[idx] = nn.Parameter(w_[row])
                try:
                    net._modules[layer].bias[idx].fill_(bias[row, col])
                except:
                    pass
                train_loss = new_loss
            else:
                fail_counts += 1
            del w_, loss_group, bias
            local_iters += 1

    return min(new_loss, train_loss)


def update_conv_weight(projection, global_bias, scd_args):
    n_w = projection.size(1)
    p_t = projection.transpose(0, 1)
    p_t = p_t.reshape((n_w, -1))
    bias_candidates = []
    num_subset_bias = []
    interval_candidates = []
    for i in range(p_t.size(0)):
        if scd_args.percentile:
            sorted_pti = p_t[i].sort()[0]
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            unique_p2 = torch.from_numpy(
                np.quantile(sorted_pti.cpu(), percentile)).type_as(p_t)
            subset_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        else:
            unique_p2 = torch.unique(p_t[i], sorted=True)
            temp_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

            if temp_bias.size(0) < scd_args.width:
                subset_bias = temp_bias
            else:
                percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
                subset_bias = torch.from_numpy(
                    np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
        bias_candidates.append(subset_bias)
        num_subset_bias.append(subset_bias.size(0))
        intervals = (subset_bias - global_bias).abs().argsort()
        interval_candidates.append(intervals)

    del p_t
    neighbor = min(scd_args.interval, min(num_subset_bias))
    bias_candidates = torch.stack(
        [subset[interval_candidates[i][:neighbor]]
         for i, subset in enumerate(bias_candidates)], dim=0)
    new_projection = projection.unsqueeze(dim=2) + \
                     bias_candidates.reshape((1, projection.size(1), neighbor, 1, 1))

    return new_projection, bias_candidates


def init_conv_diversity(net, data, layer, criterion, target, dtype, idx, scd_args):
    p2 = net(data, input_=layer, layer=layer + '_projection')
    n_nodes = p2.size(1)
    height = p2.size(2)
    width = p2.size(3)
    if not scd_args.percentile:
        unique_p2 = torch.unique(p2[:, idx], sorted=True).to(dtype=dtype)
        temp_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

        if temp_bias.size(0) < scd_args.width:
            subset_bias = temp_bias
        else:
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            subset_bias = torch.from_numpy(
                np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
    else:
        sorted_pti = p2[:, idx].flatten().sort()[0]
        percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
        unique_p2 = torch.from_numpy(
            np.quantile(sorted_pti.cpu(), percentile)).type_as(p2)
        subset_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        del unique_p2

    new_projection = (p2[:, idx:idx + 1] + subset_bias.reshape((1, -1, 1, 1))).sign_()
    # new_projection shape nrows(1500) * n_bias(100) * h(32) * w(32)
    np0 = new_projection[target == 0]
    np1 = new_projection[target == 1]
    del new_projection
    match_rate = (np0 == np1).float()
    match_rate = match_rate.view((match_rate.size(0), match_rate.size(1), -1)).mean(dim=-1).mean(dim=0)
    best_index = match_rate.argmin()
    best_bias = subset_bias[best_index]

    net._modules[layer].bias[idx].fill_(best_bias)
    return match_rate[best_index], best_bias


def update_conv_weight_for_diversity(projection, global_bias, target,
                scd_args):
    n_w = projection.size(1)
    p_t = projection.transpose(0, 1)
    p_t = p_t.reshape((n_w, -1))
    bias_candidates = []
    num_subset_bias = []
    interval_candidates = []
    for i in range(p_t.size(0)):
        if scd_args.percentile:
            sorted_pti = p_t[i].sort()[0]
            percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
            unique_p2 = torch.from_numpy(
                np.quantile(sorted_pti.cpu(), percentile)).type_as(p_t)
            subset_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        else:
            unique_p2 = torch.unique(p_t[i], sorted=True)
            temp_bias = -1.0 * (
                    unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2

            if temp_bias.size(0) < scd_args.width:
                subset_bias = temp_bias
            else:
                percentile = 1.0 * np.arange(scd_args.width) / scd_args.width
                subset_bias = torch.from_numpy(
                    np.quantile(temp_bias.cpu(), percentile)).type_as(temp_bias)
        del unique_p2
        bias_candidates.append(subset_bias)
        num_subset_bias.append(subset_bias.size(0))
        intervals = (subset_bias - global_bias).abs().argsort()
        interval_candidates.append(intervals)

    del p_t
    neighbor = min(scd_args.interval, min(num_subset_bias))
    bias_candidates = torch.stack(
        [subset[interval_candidates[i][:neighbor]]
         for i, subset in enumerate(bias_candidates)], dim=0)
    new_projection = (projection.unsqueeze_(dim=2) + \
                     bias_candidates.reshape((1, n_w, neighbor, 1, 1))).sign_().char()
    np0 = new_projection[target == 0]
    np1 = new_projection[target == 1]
    del new_projection, projection

    match_rate = (np0 == np1).float()
    match_rate = match_rate.view((match_rate.size(0), match_rate.size(1), match_rate.size(2), -1)).mean(dim=-1).mean(dim=0)
    row, col = np.unravel_index(match_rate.argmin().cpu().numpy(), match_rate.shape)

    return match_rate[row, col], row, bias_candidates[row, col]

def update_conv_diversity(net, layer, layers, layer_index, data, dtype, scd_args,
                          criterion, target, device, idx):
    net._modules[layer].bias[idx].zero_()
    # Get the global bias for this batch
    train_match_rate, global_bias = init_conv_diversity(net, data, layer, criterion,
                                                        target, dtype, idx, scd_args)

    weights = net._modules[layer].weight
    weight_size = weights.size()[1:]
    n_nodes = weight_size[0] * weight_size[1] * weight_size[2]
    best_w = weights[idx:idx + 1].clone()
    w_incs1 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc1
    local_iters = 0
    fail_counts = 0
    while local_iters < scd_args.local_iter and \
            fail_counts < scd_args.fail_count:

        updated_features = min(n_nodes, scd_args.updated_conv_features_diversity)
        cords_index = np.random.choice(n_nodes, updated_features, False)
        cords = []
        for i in range(weight_size[0]):
            for j in range(weight_size[1]):
                for k in range(weight_size[2]):
                    cords.append([i, j, k])
        cords = torch.tensor(cords)[cords_index]

        if 'si' in layer or layer_index == len(layers) - 1 or 'sign' not in scd_args.act:
            inc = []
            for j in range(w_incs1.shape[0]):
                w_inc = w_incs1[j]
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                for i in range(updated_features):
                    w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
                inc.append(w_)
            w_ = torch.cat(inc, dim=0)
            del inc
            if scd_args.normalize:
                w_ /= w_.view((updated_features * w_incs1.shape[0], -1)).norm(dim=1).view(
                    (-1, 1, 1, 1))
            # w_ = torch.cat([w_, -1.0 * w_], dim=1)
        else:
            w_incs2 = -1

            w_ = torch.repeat_interleave(
                best_w, updated_features, dim=0)
            for i in range(updated_features):
                w_[i, cords[i][0], cords[i][1], cords[i][2]] *= w_incs2

        ic = w_.shape[0]

        temp_module = torch.nn.Conv2d(in_channels=data.size(1), out_channels=ic, stride=net._modules[layer].stride,
                                      kernel_size=list(weights.size()[2:]),
                                      padding=net._modules[layer].padding).to(dtype=dtype,
                                                                              device=device)
        temp_module.weight = nn.Parameter(w_)
        temp_module.bias.zero_()
        temp_module.requires_grad_(False)

        # projection's shape  nrows(1500) * ic(96) * H * W

        projection = temp_module(data)
        del temp_module
        temp_match_rate, row, temp_bias = update_conv_weight_for_diversity(
            projection, global_bias, target, scd_args)

        if temp_match_rate < train_match_rate:
            train_match_rate = temp_match_rate
            global_bias = temp_bias.clone()
            net._modules[layer].weight[idx] = nn.Parameter(w_[row])
            net._modules[layer].bias[idx].fill_(global_bias)
        else:
            fail_counts += 1
        del w_, temp_bias
        local_iters += 1
    print(f'{layer} {idx} nodes match rate: {train_match_rate}')
    return train_match_rate


def update_mid_layer_conv_diversity(net, layers, layer_index, data_, dtype, scd_args,
                          criterion, target, device):
    layer = layers[layer_index]
    if layer_index == len(layers) - 1:
        data = data_
    else:
        previous_layer = layers[layer_index + 1]
        data = net(data_, input_=layers[-1], layer=previous_layer + '_output')

    num_nodes = net._modules[layer].weight.shape[0]
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:min(num_nodes, scd_args.updated_conv_nodes)]:
        train_match_rate = update_conv_diversity(net, layer, layers, layer_index, data, dtype, scd_args,
                          criterion, target, device, idx)

    return train_match_rate


# @profile
# def update_mid_layer_conv_nobias(net, layers, layer_index, data_, dtype, scd_args,
#                           criterion, target, device):
#     layer = layers[layer_index]
#     if layer_index == len(layers) - 1:
#         data = data_
#     else:
#         previous_layer = layers[layer_index + 1]
#         data = net(data_, input_=layers[-1], layer=previous_layer + '_output')
#
#     num_nodes = net._modules[layer].weight.shape[0]
#     for idx in np.random.permutation(
#             net._modules[layer].weight.shape[0])[:min(num_nodes, scd_args.updated_conv_nodes)]:
#         if scd_args.conv_diversity:
#             if scd_args.diversity:
#                 idx = get_diversity_index(data, layer, net, target)
#             train_match_rate = update_conv_diversity(net, layer, layers, layer_index, data, dtype,
#                                                  scd_args,
#                                                  criterion, target, device, idx)
#
#         net._modules[layer].bias[idx].zero_()
#         # Get the global bias for this batch
#         if scd_args.updated_conv_ratio > 1:
#             batch_size = data.size(0) // scd_args.updated_conv_ratio
#             rest = data.size(0) % scd_args.updated_conv_ratio
#             yps = []
#
#             for batch_idx in range(scd_args.updated_conv_ratio):
#                 yps.append(net(data[batch_size * batch_idx: batch_size * (batch_idx + 1)], input_=layer))
#             if rest:
#                 yps.append(net(data[batch_size * scd_args.updated_conv_ratio:], input_=layer))
#             yp = torch.cat(yps, dim=0)
#         else:
#             yp = net(data, input_=layer)
#         train_loss = criterion(yp, target)
#
#         weights = net._modules[layer].weight
#         weight_size = weights.size()[1:]
#         n_nodes = weight_size[0] * weight_size[1] * weight_size[2]
#         best_w = weights[idx:idx + 1].clone()
#         w_incs1 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc1
#         local_iters = 0
#         fail_counts = 0
#         while local_iters < scd_args.local_iter and \
#                 fail_counts < scd_args.fail_count:
#
#             updated_features = min(n_nodes, scd_args.updated_conv_features)
#             cords_index = np.random.choice(n_nodes, updated_features, False)
#             cords = []
#             for i in range(weight_size[0]):
#                 for j in range(weight_size[1]):
#                     for k in range(weight_size[2]):
#                         cords.append([i, j, k])
#             cords = torch.tensor(cords)[cords_index]
#
#             inc = []
#             for j in range(w_incs1.shape[0]):
#                 w_inc = w_incs1[j]
#                 w_ = torch.repeat_interleave(
#                     best_w, updated_features, dim=0)
#                 for i in range(updated_features):
#                     w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
#                 inc.append(w_)
#             w_ = torch.cat(inc, dim=0)
#             del inc
#             if scd_args.normalize:
#                 w_ /= w_.view((updated_features * w_incs1.shape[0], -1)).norm(dim=1).view(
#                     (-1, 1, 1, 1))
#
#             ic = w_.shape[0]
#
#             temp_module = torch.nn.Conv2d(in_channels=data.size(1), out_channels=ic, stride=net._modules[layer].stride,
#                                           kernel_size=list(weights.size()[2:]),
#                                           padding=net._modules[layer].padding).to(dtype=dtype,
#                                                                                   device=device)
#             temp_module.weight = nn.Parameter(w_)
#             temp_module.bias.zero_()
#             temp_module.requires_grad_(False)
#
#             # projection's shape  nrows(1500) * ic(96) * H * W
#
#             # del temp_module
#             batch_size = data.size(0) // scd_args.updated_conv_ratio
#             rest = data.size(0) % scd_args.updated_conv_ratio
#             yps = []
#
#             for batch_idx in range(scd_args.updated_conv_ratio):
#                 new_projection = temp_module(data[batch_size * batch_idx: (batch_idx + 1) * batch_size])
#
#                 n_r = new_projection.size(0)  # 1500
#                 n_w = new_projection.size(1)  # 6
#                 height = new_projection.size(2)  # 32
#                 width = new_projection.size(3)
#                 # new_projection 1500*16*20  bias 16*20
#                 # original projection feed into next layer
#                 projection = net(data[batch_size * batch_idx: (batch_idx + 1) * batch_size],
#                                  input_=layer, layer=layer + '_projection')
#                 projection.unsqueeze_(dim=1)
#                 # replace projection[:, idx] after flatten variations
#                 projection = torch.repeat_interleave(projection, n_w, dim=1)
#                 projection[:, :, idx] = new_projection
#                 del new_projection
#                 projection = projection.transpose_(0, 1).reshape(
#                     (-1, projection.size(2), height, width))
#
#                 yp = net(projection, input_=layer + '_ap').reshape((n_w, n_r, -1))
#                 del projection
#                 yp = yp.transpose_(0, 1).reshape((n_r, n_w,  -1))
#                 yps.append(yp)
#
#             if rest > 0:
#                 new_projection = temp_module(data[batch_size * scd_args.updated_conv_ratio:])
#
#                 n_r = new_projection.size(0)  # 1500
#                 n_w = new_projection.size(1)  # 6
#                 height = new_projection.size(2)  # 32
#                 width = new_projection.size(3)
#                 # new_projection 1500*16*20  bias 16*20
#                 # original projection feed into next layer
#                 projection = net(data[batch_size * scd_args.updated_conv_ratio:],
#                                  input_=layer, layer=layer + '_projection')
#                 projection.unsqueeze_(dim=1)
#                 # replace projection[:, idx] after flatten variations
#                 projection = torch.repeat_interleave(projection, n_w, dim=1)
#                 projection[:, :, idx] = new_projection
#                 del new_projection
#                 projection = projection.transpose_(0, 1).reshape(
#                     (-1, projection.size(2), height, width))
#                 yp = net(projection, input_=layer + '_ap').reshape((n_w, n_r, -1))
#                 del projection
#                 yp = yp.transpose_(0, 1).reshape((n_r, n_w,  -1))
#                 yps.append(yp)
#
#             yp = torch.cat(yps, dim=0)
#             loss_group = criterion(yp, target)
#             # new_loss = loss_group.min()
#             # if new_loss <= train_loss:
#             #     row = loss_group.argmin()
#             #     net._modules[layer].weight[idx] = nn.Parameter(w_[row], requires_grad=False)
#             #     # net._modules[layer].bias[idx].fill_(bias[row, col])
#             #     train_loss = new_loss
#             # else:
#             #     fail_counts += 1
#             update_idx = loss_group < train_loss
#             num_updates = update_idx.sum()
#             if num_updates:
#                 # print(f'{layer} valid update: {num_updates}')
#                 row = loss_group.argmin()
#                 net._modules[layer].weight[idx] = nn.Parameter(w_[row], requires_grad=False)
#                 train_loss = loss_group.min()
#             else:
#                 fail_counts += 1
#             del w_, loss_group
#             local_iters += 1
#
#     return train_loss


def update_mid_layer_conv_nobias(net, layers, layer_index, data_, dtype, scd_args,
                          criterion, target, device, bnn=False):
    layer = layers[layer_index]
    if layer_index == len(layers) - 1:
        data = data_
    else:
        previous_layer = layers[layer_index + 1]
        data = net(data_, input_=layers[-1], layer=previous_layer + '_output')

    num_nodes = net._modules[layer].weight.shape[0]
    for idx in np.random.permutation(
            net._modules[layer].weight.shape[0])[:min(num_nodes, scd_args.updated_conv_nodes)]:
        if scd_args.conv_diversity:
            if scd_args.diversity:
                idx = get_diversity_index(data, layer, net, target)
            train_match_rate = update_conv_diversity(net, layer, layers, layer_index, data, dtype,
                                                 scd_args,
                                                 criterion, target, device, idx)
        try:
            net._modules[layer].bias[idx].zero_()
        except:
            pass
        # Get the global bias for this batch
        if scd_args.updated_conv_ratio > 1:
            batch_size = data.size(0) // scd_args.updated_conv_ratio
            rest = data.size(0) % scd_args.updated_conv_ratio
            yps = []

            for batch_idx in range(scd_args.updated_conv_ratio):
                yps.append(net(data[batch_size * batch_idx: batch_size * (batch_idx + 1)],
                               input_=layer, layer=net.layers[-1]+'_projection'))
            if rest:
                yps.append(net(data[batch_size * scd_args.updated_conv_ratio:],
                               input_=layer, layer=net.layers[-1]+'_projection'))
            yp = torch.cat(yps, dim=0)
        else:
            yp = net(data, input_=layer)
        train_loss = criterion(yp, target)

        # if layer is under bnn training, binarize weights
        if bnn:
            net._modules[layer].weight.data.sign_()
        weights = net._modules[layer].weight
        weight_size = weights.size()[1:]
        n_nodes = weight_size[0] * weight_size[1] * weight_size[2]
        best_w = weights[idx:idx + 1].clone()


        w_incs1 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc1
        local_iters = 0
        fail_counts = 0
        while local_iters < scd_args.local_iter and \
                fail_counts < scd_args.fail_count:

            updated_features = min(n_nodes, scd_args.updated_conv_features)
            cords_index = np.random.choice(n_nodes, updated_features, False)
            cords = []
            for i in range(weight_size[0]):
                for j in range(weight_size[1]):
                    for k in range(weight_size[2]):
                        cords.append([i, j, k])
            cords = torch.tensor(cords)[cords_index]

            if bnn:
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                for i in range(updated_features):
                    w_[i, cords[i][0], cords[i][1], cords[i][2]] *= -1
            else:
                inc = []
                for j in range(w_incs1.shape[0]):
                    w_inc = w_incs1[j]
                    w_ = torch.repeat_interleave(
                        best_w, updated_features, dim=0)
                    for i in range(updated_features):
                        w_[i, cords[i][0], cords[i][1], cords[i][2]] += w_inc
                    inc.append(w_)
                w_ = torch.cat(inc, dim=0)
                del inc
                if scd_args.normalize:
                    w_ /= w_.view((updated_features * w_incs1.shape[0], -1)).norm(dim=1).view(
                        (-1, 1, 1, 1))

            ic = w_.shape[0]

            temp_module = torch.nn.Conv2d(in_channels=data.size(1), out_channels=ic, stride=net._modules[layer].stride,
                                          kernel_size=list(weights.size()[2:]),
                                          padding=net._modules[layer].padding).to(dtype=dtype,
                                                                                  device=device)
            temp_module.weight = nn.Parameter(w_)
            temp_module.bias.zero_()
            temp_module.requires_grad_(False)

            # projection's shape  nrows(1500) * ic(96) * H * W

            # del temp_module
            batch_size = data.size(0) // scd_args.updated_conv_ratio
            rest = data.size(0) % scd_args.updated_conv_ratio
            yps = []

            for batch_idx in range(scd_args.updated_conv_ratio):
                new_projection = temp_module(data[batch_size * batch_idx: (batch_idx + 1) * batch_size])

                n_r = new_projection.size(0)  # 1500
                n_w = new_projection.size(1)  # 6
                height = new_projection.size(2)  # 32
                width = new_projection.size(3)
                # new_projection 1500*16*20  bias 16*20
                # original projection feed into next layer
                projection = net(data[batch_size * batch_idx: (batch_idx + 1) * batch_size],
                                 input_=layer, layer=layer + '_projection')
                projection.unsqueeze_(dim=1)
                # replace projection[:, idx] after flatten variations
                projection = torch.repeat_interleave(projection, n_w, dim=1)
                projection[:, :, idx] = new_projection
                del new_projection
                projection = projection.transpose_(0, 1).reshape(
                    (-1, projection.size(2), height, width))

                yp = net(projection, input_=layer + '_ap').reshape((n_w, n_r, -1))
                del projection
                yp = yp.transpose_(0, 1).reshape((n_r, n_w,  -1))
                yps.append(yp)

            if rest > 0:
                new_projection = temp_module(data[batch_size * scd_args.updated_conv_ratio:])

                n_r = new_projection.size(0)  # 1500
                n_w = new_projection.size(1)  # 6
                height = new_projection.size(2)  # 32
                width = new_projection.size(3)
                # new_projection 1500*16*20  bias 16*20
                # original projection feed into next layer
                projection = net(data[batch_size * scd_args.updated_conv_ratio:],
                                 input_=layer, layer=layer + '_projection')
                projection.unsqueeze_(dim=1)
                # replace projection[:, idx] after flatten variations
                projection = torch.repeat_interleave(projection, n_w, dim=1)
                projection[:, :, idx] = new_projection
                del new_projection
                projection = projection.transpose_(0, 1).reshape(
                    (-1, projection.size(2), height, width))
                yp = net(projection, input_=layer + '_ap',
                         layer=net.layers[-1]+'_projection').reshape((n_w, n_r, -1))
                del projection
                yp = yp.transpose_(0, 1).reshape((n_r, n_w,  -1))
                yps.append(yp)

            yp = torch.cat(yps, dim=0)
            loss_group = criterion(yp, target)
            # new_loss = loss_group.min()
            # if new_loss <= train_loss:
            #     row = loss_group.argmin()
            #     net._modules[layer].weight[idx] = nn.Parameter(w_[row], requires_grad=False)
            #     # net._modules[layer].bias[idx].fill_(bias[row, col])
            #     train_loss = new_loss
            # else:
            #     fail_counts += 1
            update_idx = loss_group < train_loss
            num_updates = update_idx.sum()
            if num_updates:
                # print(f'{layer} valid update: {num_updates}')
                row = loss_group.argmin()
                net._modules[layer].weight[idx] = nn.Parameter(w_[row])
                train_loss = loss_group.min()
            else:
                fail_counts += 1
            del w_, loss_group
            local_iters += 1

    return train_loss
