import torch


def update_bp_ce(net, optimizer, data_loader, criterion, use_cuda, device, dtype,
                 attacker=None):
    net.train()
    for data, target in data_loader:
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        if attacker is not None:
            adv_data = attacker.perturb(data, target, 'mean', True)
            adv_data = adv_data.type_as(data)
            data = torch.cat([data, adv_data], dim=0)
            target = torch.cat([target, target], dim=0)

        optimizer.zero_grad()
        outputs = net(data, layer=net.layers[-1] + '_projection')
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        break


def update_bp_bce(net, optimizer, data_loader, criterion, use_cuda, device, dtype,
                  attacker=None):
    net.train()
    for data, target in data_loader:
        if use_cuda:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        if attacker is not None:
            adv_data = attacker.perturb(data, target, 'mean', True)
            adv_data = adv_data.type_as(data)
            data = torch.cat([data, adv_data], dim=0)
            target = torch.cat([target, target], dim=0)

        optimizer.zero_grad()
        outputs = net(data, layer=net.layers[-1] + '_projection').flatten()
        loss = criterion(outputs, target.type_as(outputs))
        loss.backward()
        optimizer.step()
        break
