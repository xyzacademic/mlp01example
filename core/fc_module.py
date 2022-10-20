import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def update_fc_weight(projection, global_bias, local_bias_idx, scd_args):
    # projection = temp_module(data.unsqueeze(dim=1))
    p_t = projection.transpose(0, 1).squeeze(dim=2)
    bias_candidates = []
    min_len = 2 * scd_args.interval
    for i in range(p_t.size(0)):
        unique_p2 = torch.unique(p_t[i], sorted=True)
        temp_bias = -1.0 * (
                unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
        if temp_bias.size(0) < 2 * scd_args.interval:
            if temp_bias.size(0) < min_len:
                min_len = temp_bias.size(0)
            min_idx = 0
            max_idx = temp_bias.size(0)
        elif local_bias_idx > (temp_bias.shape[0] - scd_args.interval):
            min_idx = temp_bias.shape[0] - 2 * scd_args.interval
            max_idx = temp_bias.shape[0]
        elif (local_bias_idx - scd_args.interval) < 0:
            min_idx = 0
            max_idx = 2 * scd_args.interval
        else:
            min_idx = local_bias_idx - scd_args.interval
            max_idx = local_bias_idx + scd_args.interval
        # print(min_idx, max_idx)
        bias_candidates.append(temp_bias[min_idx: max_idx])
        # print(i)
    if min_len < 2 * scd_args.interval:
        for i, temp_bias in enumerate(bias_candidates):
            bias_candidates[i] = temp_bias[:min_len]

    del p_t
    bias_candidates = torch.stack(bias_candidates, dim=0)

    return projection + bias_candidates.unsqueeze(dim=0), bias_candidates


def init_fc(net, data, layer, criterion, target, dtype, idx, scd_args):
    p2 = net(data, input_=layer, layer=layer + '_projection')
    nrows = p2.size(0)
    n_nodes = p2.size(1)
    # p2 shape nrows(1500) * nodes(20)
    unique_p2 = torch.unique(p2[:, idx], sorted=True).to(dtype=dtype)
    temp_bias = -1.0 * (
            unique_p2 + torch.cat([unique_p2[0:1] - 0.1, unique_p2[:-1]])) / 2
    n_variations = temp_bias.size(0)
    del unique_p2

    batch_size = nrows // scd_args.updated_fc_ratio
    rest = p2.size(0) % scd_args.updated_fc_ratio
    yps = []
    for batch_i in range(scd_args.updated_fc_ratio):
        new_projection = p2[batch_i * batch_size: (batch_i+1) * batch_size,
                         idx:idx + 1] + temp_bias.unsqueeze(dim=0)
        nrows = new_projection.size(0)
        # new_projection shape nrows(1500) * n_bias(508)
        new_p2 = torch.repeat_interleave(
            p2[batch_i * batch_size: (batch_i+1) * batch_size
                    ].unsqueeze(dim=1), n_variations, dim=1)  # nrows * n_bias * nodes
        # del p2
        # replace original idx's projection by variations with different bias
        new_p2[:, :, idx] = new_projection  # n_rows, n_bias, n_nodes
        del new_projection
        # switch dimension of nrows and n_bias, then flatten these two dimension
        new_projection = new_p2.transpose_(0, 1).reshape((-1, n_nodes))  # n_bias * nrows * nodes
        del new_p2
        # get the final output, and reverse to original dimension order  nrows * n_bias * 1
        try:
            yp = net(new_projection, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape(
                (n_variations, nrows, -1)).transpose_(0, 1)
        except:
            yp = new_projection.reshape(
                (n_variations, nrows, -1)).transpose_(0, 1)
        del new_projection
        yps.append(yp)

    if rest > 0:
        new_projection = p2[scd_args.updated_fc_ratio * batch_size:,
                         idx:idx + 1] + temp_bias.unsqueeze(dim=0)
        nrows = new_projection.size(0)
        # new_projection shape nrows(1500) * n_bias(508)
        new_p2 = torch.repeat_interleave(
            p2[scd_args.updated_fc_ratio * batch_size:
               ].unsqueeze(dim=1), n_variations, dim=1)  # nrows * n_bias * nodes
        # del p2
        # replace original idx's projection by variations with different bias
        new_p2[:, :, idx] = new_projection  # n_rows, n_bias, n_nodes
        del new_projection
        # switch dimension of nrows and n_bias, then flatten these two dimension
        new_projection = new_p2.transpose_(0, 1).reshape((-1, n_nodes))  # n_bias * nrows * nodes
        del new_p2
        # get the final output, and reverse to original dimension order  nrows * n_bias * 1
        try:
            yp = net(new_projection, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape(
                (n_variations, nrows, -1)).transpose_(0, 1)
        except:
            yp = new_projection.reshape(
                (n_variations, nrows, -1)).transpose_(0, 1)
        del new_projection
        yps.append(yp)

    yp = torch.cat(yps, dim=0)
    # loss_group = criterion(yp, torch.stack([target for i in range(yp.size(1))], dim=1).type_as(yp).unsqueeze(dim=-1))
    # loss_group = loss_group.mean(dim=0).squeeze()
    loss_group = criterion(yp, target)
    best_index = loss_group.argmin()
    best_bias = temp_bias[best_index]
    try:
        net._modules[layer].bias[idx].fill_(best_bias)
    except:
        pass
    if scd_args.record:
        loss_name = f'{layer}_loss'
        bias_name = f'{layer}_bias'
        if loss_name not in scd_args.logs:
            scd_args.logs[loss_name] = []
        if bias_name not in scd_args.logs:
            scd_args.logs[bias_name] = []
        scd_args.logs[loss_name].append(loss_group.cpu().numpy())
        scd_args.logs[bias_name].append(temp_bias.cpu().numpy())
    del temp_bias
    return loss_group[best_index], best_bias, best_index.item()


def update_mid_layer_fc(net, layers, layer_index, data_, dtype, scd_args, criterion, target,
                        device, bnn=False):
    layer = layers[layer_index]
    if layer_index == len(layers) - 1:
        data = data_
    else:
        previous_layer = layers[layer_index + 1]
        data = net(data_, input_=layers[-1], layer=previous_layer + '_output')

    num_nodes = net._modules[layer].weight.shape[0]
    for idx in np.random.permutation(
            num_nodes)[:min(num_nodes, scd_args.updated_fc_nodes)]:
        try:
            net._modules[layer].bias[idx].zero_()
        except:
            pass
        train_loss, global_bias, best_index = init_fc(net, data, layer, criterion,
                                          target, dtype, idx, scd_args)

        if bnn:
            net._modules[layer].weight.data.sign_()
        weights = net._modules[layer].weight
        num_input_features = weights.size(1)  # get input's dimension
        updated_features = min(num_input_features, scd_args.updated_fc_features)

        best_w = weights[idx:idx + 1].clone()
        w_incs2 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc2
        # shuffle updating order
        local_iters = 0
        fail_counts = 0
        while local_iters < scd_args.local_iter and \
                fail_counts < scd_args.fail_count:

            cords = np.random.choice(num_input_features, updated_features, False)
            # if 'si' in layer or layer_index == 0 or 'sign' not in scd_args.act:

            if bnn:
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                w_[np.arange(updated_features), cords] *= -1
            else:
                inc = []
                for i in range(w_incs2.shape[0]):
                    w_inc = w_incs2[i]
                    w_ = torch.repeat_interleave(
                        best_w, updated_features, dim=0)
                    w_[np.arange(updated_features), cords] += w_inc
                    inc.append(w_)
                w_ = torch.cat(inc, dim=0)
                del inc
                if scd_args.normalize:
                    w_ /= w_.norm(dim=1, keepdim=True)

            ic = w_.shape[0]

            temp_module = torch.nn.Conv1d(in_channels=1, out_channels=ic, stride=1,
                                          kernel_size=weights.size(1)).to(dtype=dtype,
                                                                          device=device)
            temp_module.weight = nn.Parameter(w_.unsqueeze(dim=1))
            temp_module.bias.zero_()
            temp_module.requires_grad_(False)

            # projection's shape  nrows(1500) * ic(12) * 1

            # del temp_module


            projection = temp_module(data.unsqueeze(dim=1))
            new_projection, bias = update_fc_weight(
                projection, global_bias, best_index,
                scd_args)
            del temp_module

            n_w = new_projection.size(1)  # 12
            n_b = new_projection.size(2)  # 20
            new_projection = new_projection.reshape(
                (new_projection.size(0), n_w * n_b))
            # new_projection 1500*12*20  bias 12*20
            # original projection feed into next layer
            projection = net(data, input_=layer,
                             layer=layer + '_projection')  # 1500 * 20
            projection.unsqueeze_(dim=1)

            # batch
            batch_size = projection.size(0) // scd_args.updated_fc_ratio
            rest = projection.size(0) % batch_size * scd_args.updated_fc_ratio
            yps = []
            # replace projection[:, idx] after flatten variations
            for i in range(scd_args.updated_fc_ratio):
                projection_batch = torch.repeat_interleave(
                    projection[batch_size * i: batch_size * (i+1)],
                    n_w * n_b, dim=1)

                projection_batch[:, :, idx] = new_projection[batch_size * i: batch_size * (i+1)]
                n_r = projection_batch.size(0)
                projection_batch = projection_batch.transpose_(0, 1).reshape(
                    (-1, projection_batch.size(2)))
                  # 1500
                try:
                    yp = net(projection_batch, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape((n_w * n_b, n_r, -1))
                except:
                    yp = projection_batch.reshape((n_w * n_b, n_r, -1))
                yp = yp.transpose_(0, 1).reshape((n_r, n_w, n_b, -1))
                yps.append(yp)

            if rest:
                projection_batch = torch.repeat_interleave(
                    projection[batch_size * scd_args.updated_fc_ratio:],
                    n_w * n_b, dim=1)

                projection_batch[:, :, idx] = new_projection[batch_size * scd_args.updated_fc_ratio:]
                n_r = projection_batch.size(0)
                projection_batch = projection_batch.transpose_(0, 1).reshape(
                    (-1, projection_batch.size(2)))
                  # 1500
                try:
                    yp = net(projection_batch, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape((n_w * n_b, n_r, -1))
                except:
                    yp = projection_batch.reshape((n_w * n_b, n_r, -1))
                yp = yp.transpose_(0, 1).reshape((n_r, n_w, n_b, -1))
                yps.append(yp)
            del projection, projection_batch
            yp = torch.cat(yps, dim=0)
            loss_group = criterion(yp, target)

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


    return train_loss

# @profile
def update_mid_layer_fc_nobias(net, layers, layer_index, data_, dtype, scd_args, criterion, target,
                        device, bnn=False):
    layer = layers[layer_index]
    if layer_index == len(layers) - 1:
        data = data_
    else:
        previous_layer = layers[layer_index + 1]
        data = net(data_, input_=layers[-1], layer=previous_layer + '_output')

    num_nodes = net._modules[layer].weight.shape[0]
    for idx in np.random.permutation(
            num_nodes)[:min(num_nodes, scd_args.updated_fc_nodes)]:
        try:
            net._modules[layer].bias[idx].zero_()
        except:
            pass
        train_loss = criterion(net(data, input_=layer, layer=net.layers[-1]+'_projection'), target)

        if bnn:
            net._modules[layer].weight.data.sign_()

        weights = net._modules[layer].weight
        num_input_features = weights.size(1)  # get input's dimension
        updated_features = min(num_input_features, scd_args.updated_fc_features)

        best_w = weights[idx:idx + 1].clone()
        w_incs2 = torch.tensor([-1, 1]).type_as(best_w) * scd_args.w_inc2
        # shuffle updating order
        local_iters = 0
        fail_counts = 0
        while local_iters < scd_args.local_iter and \
                fail_counts < scd_args.fail_count:

            cords = np.random.choice(num_input_features, updated_features, False)
            # if 'si' in layer or layer_index == 0 or 'sign' not in scd_args.act:
            if bnn:
                w_ = torch.repeat_interleave(
                    best_w, updated_features, dim=0)
                w_[np.arange(updated_features), cords] *= -1

            else:
                inc = []
                for i in range(w_incs2.shape[0]):
                    w_inc = w_incs2[i]
                    w_ = torch.repeat_interleave(
                        best_w, updated_features, dim=0)
                    w_[np.arange(updated_features), cords] += w_inc
                    inc.append(w_)
                w_ = torch.cat(inc, dim=0)
                del inc
                if scd_args.normalize:
                    w_ /= w_.norm(dim=1, keepdim=True)

            ic = w_.shape[0]

            temp_module = torch.nn.Conv1d(in_channels=1, out_channels=ic, stride=1,
                                          kernel_size=weights.size(1)).to(dtype=dtype,
                                                                          device=device)
            temp_module.weight = nn.Parameter(w_.unsqueeze(dim=1))
            temp_module.bias.zero_()
            temp_module.requires_grad_(False)

            # projection's shape  nrows(1500) * ic(12) * 1

            # del temp_module
            new_projection = temp_module(data.unsqueeze(dim=1))
            del temp_module
            n_r = new_projection.size(0)  # 1500
            n_w = new_projection.size(1)  # 12

            # new_projection 1500*12*20  bias 12*20
            # original projection feed into next layer
            projection = net(data,
                             input_=layer, layer=layer + '_projection')  # 1500 * 20
            projection.unsqueeze_(dim=1)
            # batch
            batch_size = projection.size(0) // scd_args.updated_fc_ratio
            rest = projection.size(0) % batch_size * scd_args.updated_fc_ratio
            yps = []
            # replace projection[:, idx] after flatten variations
            for i in range(scd_args.updated_fc_ratio):
                projection_batch = torch.repeat_interleave(
                    projection[batch_size * i: batch_size * (i+1)], n_w, dim=1)
                projection_batch[:, :, idx] = new_projection[batch_size * i: batch_size * (i+1)].squeeze(dim=-1)
                n_r = projection_batch.size(0)
                projection_batch = projection_batch.transpose_(0, 1).reshape(
                    (-1, projection_batch.size(2)))
                try:
                    yp = net(projection_batch, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape((n_w, n_r, -1))
                except:
                    yp = projection_batch.reshape((n_w, n_r, -1))
                yp = yp.transpose_(0, 1).reshape((n_r, n_w, -1))
                yps.append(yp)

            if rest:
                projection_batch = torch.repeat_interleave(
                    projection[batch_size * scd_args.updated_fc_ratio:], n_w, dim=1)
                projection_batch[:, :, idx] = new_projection[
                                              batch_size * scd_args.updated_fc_ratio:].squeeze(dim=-1)
                n_r = projection_batch.size(0)
                projection_batch = projection_batch.transpose_(0, 1).reshape(
                    (-1, projection_batch.size(2)))
                try:
                    yp = net(projection_batch, input_=layer + '_ap', layer=net.layers[-1]+'_projection').reshape((n_w, n_r, -1))
                except:
                    yp = projection_batch.reshape((n_w, n_r, -1))
                yp = yp.transpose_(0, 1).reshape((n_r, n_w, -1))
                yps.append(yp)

            del projection, projection_batch
            yp = torch.cat(yps, dim=0)
            loss_group = criterion(yp, target)

            new_loss = loss_group.min()
            if new_loss <= train_loss:
                row = loss_group.argmin()
                net._modules[layer].weight[idx] = nn.Parameter(w_[row])
                train_loss = new_loss
            else:
                fail_counts += 1
            del w_, loss_group
            local_iters += 1


    return train_loss
