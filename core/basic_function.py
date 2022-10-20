import time
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
import torch


def evaluation(data_loader, use_cuda, device, dtype, net, key, criterion, attacker=None):
    a = time.time()
    pred = []
    labels = []
    net.eval()
    yps = []
    label = []
    yps_adv = []
    pred_adv = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.type_as(net._modules[list(net._modules.keys())[0]].weight), target.to(device=device)
            if attacker is not None:
                adv_data = attacker.perturb(data, target, 'mean', False)
                adv_data = adv_data.type_as(data)
                yp_adv = net(adv_data, layer=net.layers[-1]+'_projection')
                yps_adv.append(yp_adv)
                if yp_adv.size(1) == 1:
                    outputs_adv = net.signb(yp_adv).round().flatten()
                else:
                    outputs_adv = yp_adv.argmax(dim=1)
                pred_adv.append(outputs_adv.cpu().numpy())
                # data = torch.cat([data, adv_data], dim=0)
                # target = torch.cat([target, target], dim=0)
            yp = net(data, layer=net.layers[-1]+'_projection')

            yps.append(yp)
            label.append(target)
            if yp.size(1) == 1:
                outputs = net.signb(yp).round().flatten()
            else:
                outputs = yp.argmax(dim=1)

            pred.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())
        yp = torch.cat(yps, dim=0)

        label = torch.cat(label, dim=0)
        loss = criterion(yp, label)
        loss = loss.item()
        pred = np.concatenate(pred, axis=0)
        labels = np.concatenate(labels, axis=0)
        acc = accuracy_score(labels, pred)
        balanced_acc = balanced_accuracy_score(labels, pred)
        print("{} balanced Accuracy: {:.5f}, imbalanced Accuracy: {:.5f}, loss: {:.5f} "
              "cost {:.2f} seconds".format(key, balanced_acc, acc, loss, time.time() - a))
        if attacker is not None:
            yp_adv = torch.cat(yps_adv, dim=0)
            loss_adv = criterion(yp_adv, label).item()
            pred_adv = np.concatenate(pred_adv, axis=0)
            acc_adv = accuracy_score(labels, pred_adv)
            balanced_acc_adv = balanced_accuracy_score(labels, pred_adv)
            print("{} adv balanced Accuracy: {:.5f}, adv imbalanced Accuracy: {:.5f}, loss: {:.5f} "
                  "".format(key, balanced_acc_adv, acc_adv, loss_adv))

            return (acc + acc_adv) / 2, (loss + loss_adv) / 2

    return acc, loss


def evaluation2(data_loader, use_cuda, device, dtype, net, key, criterion, attacker=None, attacker2=None):
    a = time.time()
    pred = []
    labels = []
    net.eval()
    yps = []
    label = []
    yps_adv = []
    pred_adv = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.type_as(net._modules[list(net._modules.keys())[0]].weight), target.to(device=device)

            if attacker2 is not None:
                adv_data = attacker2.perturb(data, target, 'mean', False)
                adv_data = adv_data.type_as(data)
                data = torch.cat([data, adv_data], dim=0)
                target = torch.cat([target, target], dim=0)

            if attacker is not None:
                adv_data = attacker.perturb(data, target, 'mean', False)
                adv_data = adv_data.type_as(data)
                yp_adv = net(adv_data, layer=net.layers[-1]+'_projection')
                yps_adv.append(yp_adv)
                if yp_adv.size(1) == 1:
                    outputs_adv = net.signb(yp_adv).round().flatten()
                else:
                    outputs_adv = yp_adv.argmax(dim=1)
                pred_adv.append(outputs_adv.cpu().numpy())
                # data = torch.cat([data, adv_data], dim=0)
                # target = torch.cat([target, target], dim=0)
            yp = net(data, layer=net.layers[-1]+'_projection')
            # print(data.size())
            yps.append(yp)
            label.append(target)
            if yp.size(1) == 1:
                outputs = net.signb(yp).round().flatten()
            else:
                outputs = yp.argmax(dim=1)

            pred.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())
        yp = torch.cat(yps, dim=0)

        label = torch.cat(label, dim=0)
        loss = criterion(yp, label)
        loss = loss.item()
        pred = np.concatenate(pred, axis=0)
        labels = np.concatenate(labels, axis=0)
        acc = accuracy_score(labels, pred)
        balanced_acc = balanced_accuracy_score(labels, pred)
        print("{} balanced Accuracy: {:.5f}, imbalanced Accuracy: {:.5f}, loss: {:.5f} "
              "cost {:.2f} seconds".format(key, balanced_acc, acc, loss, time.time() - a))
        if attacker is not None:
            yp_adv = torch.cat(yps_adv, dim=0)
            loss_adv = criterion(yp_adv, label).item()
            pred_adv = np.concatenate(pred_adv, axis=0)
            acc_adv = accuracy_score(labels, pred_adv)
            balanced_acc_adv = balanced_accuracy_score(labels, pred_adv)
            print("{} adv balanced Accuracy: {:.5f}, adv imbalanced Accuracy: {:.5f}, loss: {:.5f} "
                  "".format(key, balanced_acc_adv, acc_adv, loss_adv))

            return (acc + acc_adv) / 2, (loss + loss_adv) / 2

    return acc, loss


def evaluation_text(data_loader, use_cuda, device, dtype, net, key, embedding, criterion):
    a = time.time()
    pred = []
    labels = []
    net.eval()
    yps = []
    label = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            data = batch.text
            target = batch.label
            if use_cuda:
                data, target = data.to(device=device), target.to(device=device)
            data = embedding(data).unsqueeze_(dim=1)
                # data, target = data.type_as(net._modules[list(net._modules.keys())[0]].weight), target.to(device=device)

            yp = net(data, layer=net.layers[-1]+'_projection')
            yps.append(yp)
            label.append(target)

            if yp.size(1) == 1:
                outputs = net.signb(yp).round().flatten()
            else:
                outputs = yp.argmax(dim=1)

            pred.append(outputs.cpu().numpy())
            labels.append(target.cpu().numpy())

    yp = torch.cat(yps, dim=0)
    label = torch.cat(label, dim=0)
    loss = criterion(yp, label)
    loss = loss.item()
    pred = np.concatenate(pred, axis=0)
    labels = np.concatenate(labels, axis=0)
    acc = accuracy_score(labels, pred)
    balanced_acc = balanced_accuracy_score(labels, pred)
    print("{} balanced Accuracy: {:.5f}, imbalanced Accuracy: {:.5f}, loss: {:.5f} "
          "cost {:.2f} seconds".format(key, balanced_acc, acc, loss, time.time() - a))
    return acc

def get_features(data_loader, use_cuda, device, dtype, net, key):
    layers = net.layers
    features = {}
    for layer in layers:
        features[layer] = []

    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        for layer in layers:
            features[layer].append(net(data, layer=layer+'_projection').sign().cpu())

    for layer in layers:
        features[layer] = torch.cat(features[layer], dim=0)

    return features