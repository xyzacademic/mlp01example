import torch
import torch.nn as nn
import os
import math

class Layer(object):
    def __init__(self, layer_type, in_channel=1, out_channel=1,
                 kernel_size=None, padding=0, bias=True,
                 pool_size=0, reshape=False, act=None, pool_type='avg', scale=False):
        self.layer_type = layer_type
        if 'conv' in  self.layer_type.lower():
            self.name = 'conv'
        elif 'linear' in  self.layer_type.lower():
            self.name = 'fc'
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.pool_size = pool_size
        self.reshape = reshape
        self.pool_type = pool_type
        self.scale = scale
        if not act:
            self.act = 'self.act'
        elif act == 'relu':
            self.act = 'torch.relu_'
        elif act == 'sigmoid':
            self.act = 'torch.sigmoid'
        elif act == 'sign':
            self.act = 'torch.sign'



def get_code_for_cnn01(name, structure, save_path):

    indent = ' ' * 4

    a = [
        f'class {name}(nn.Module):',
        f'{indent}def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, bias=True):',
        f'{indent}{indent}super({name}, self).__init__()',
        f'{indent}{indent}if act == "sign":',
        f'{indent}{indent}{indent}self.act = torch.sign',
        f'{indent}{indent}elif act == "signb":',
        f'{indent}{indent}{indent}self.act = signb',
        f'{indent}{indent}elif act == "sigmoid":',
        f'{indent}{indent}{indent}self.act = torch.sigmoid_',
        f'{indent}{indent}elif act == "relu":',
        f'{indent}{indent}{indent}self.act = torch.relu_',
        '',
        f'{indent}{indent}if softmax:',
        f'{indent}{indent}{indent}if num_classes < 2:',
        f'{indent}{indent}{indent}{indent}' + 'raise ValueError("num_classes expect larger than 1, but got {num_classes}")',
        f'{indent}{indent}{indent}self.signb = softmax_',
        f'{indent}{indent}else:',
        f'{indent}{indent}{indent}self.signb = torch.sigmoid if sigmoid else signb',
        '',
    ]

    b = []
    names = []
    for i, layer in enumerate(structure):
        main_body = f'self.{layer.name}{i+1}_si = {layer.layer_type}'
        parameters = [
            f'{layer.in_channel}',
            f'{layer.out_channel}',
            ]
        if layer.kernel_size:
            parameters.append(f'kernel_size={layer.kernel_size}')
        if layer.padding:
            parameters.append(f'padding={layer.padding}')
        if layer.bias:
            parameters.append(f'bias=bias',)
        param = ", ".join(parameters)
        command = f'{indent}{indent}{main_body}({param})'
        b.append(command)
        names.append(f'"{layer.name}{i+1}_si"')

    layer_list = ", ".join(names)
    b.append(f'{indent}{indent}self.layers = [{layer_list}]')
    b.append(f'{indent}{indent}self.apply(_weights_init)')
    b.append('')

    c = [
        f'{indent}def forward(self, x, input_=None, layer=None):',
        f'{indent}{indent}# check input start from which layer',
        f'{indent}{indent}status = -1',
        f'{indent}{indent}for items in self.layers:',
        f'{indent}{indent}{indent}status += 1',
        f'{indent}{indent}{indent}if input_ is None or items in input_:',
        f'{indent}{indent}{indent}{indent}break',
        '',
    ]


    for i, layer in enumerate(structure):
        temp = [f'{indent}{indent}# layer {i+1}']  # initial a temp list for saving each layer's forward command
        if i > 0:
            temp += [
                f'{indent}{indent}if input_ == self.layers[{i}]:',
                f'{indent}{indent}{indent}out = x',
                     ]

        temp += [
            f'{indent}{indent}if status < {i+1}:',
            f'{indent}{indent}{indent}if input_ != self.layers[{i}] + "_ap":',
            ]
        if i == 0:
            temp += [
                f'{indent}{indent}{indent}{indent}out = self.%s(x)' % names[i].replace('"',''),
            ]
        else:
            temp += [
                f'{indent}{indent}{indent}{indent}out = self.%s(out)' % names[i].replace('"',''),
            ]
        temp += [
            f'{indent}{indent}{indent}if layer == self.layers[{i}] + "_projection":',
            f'{indent}{indent}{indent}{indent}return out',
            f'{indent}{indent}{indent}if input_ == self.layers[{i}] + "_ap":',
            f'{indent}{indent}{indent}{indent}out = x',
        ]
        if i < len(structure) - 1:
            if layer.scale:
                if layer.kernel_size:
                    scale = math.sqrt(layer.out_channel) * layer.kernel_size
                else:
                    scale = math.sqrt(layer.out_channel)
                temp += [f'{indent}{indent}{indent}out = {layer.act}(out)'+' * {:.4f}'.format(1/scale)]
            else:
                temp += [f'{indent}{indent}{indent}out = {layer.act}(out)']
        else:
            temp += [f'{indent}{indent}{indent}out = self.signb(out)']
        if layer.pool_size:
            if layer.pool_type == 'avg':
                temp += [f'{indent}{indent}{indent}out = F.avg_pool2d(out, {layer.pool_size})']
            elif layer.pool_type == 'fs':
                temp += [f'{indent}{indent}{indent}out = F.relu(out)']
                temp += [f'{indent}{indent}{indent}out = F.avg_pool2d(out, {layer.pool_size}) * {layer.pool_size ** 2}']
        if layer.reshape:
            temp += [f'{indent}{indent}{indent}out = out.reshape(out.size(0), -1)']

        if i < len(structure) - 1:
            temp += [
                f'{indent}{indent}{indent}if layer == self.layers[{i}] + "_output":',
                f'{indent}{indent}{indent}{indent}return out',
                '',
            ]
        else:
            temp += [
                '',
                f'{indent}{indent}return out',
            ]

        c += temp

    content = a + b + c

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    with open('../core/cnn01.py', 'r') as f:
        contents = f.readlines()

    id1 = contents.index('arch = {}\n')
    id2 = contents.index("if __name__ == '__main__':\n")
    part1 = contents[:id1]
    part2 = contents[id1:id2]
    part3 = contents[id2:]

    part1 += [line + '\n' for line in content]
    part1 += '\n\n'
    part2 += f"arch['{name.lower()}'] = {name}\n"

    new_contents = part1 + part2 + part3

    with open('../core/cnn01.py', 'w') as f:
        f.writelines(new_contents)


    # with open(os.path.join(save_path, f'{name}_cnn01.py'), 'w') as f:
    #     for line in contents:
    #         f.write(line + '\n')


def get_code_for_ensemble(name, structure, save_path):
    indent = ' ' * 4

    a = [
        f'class {name}(nn.Module):',
        f'{indent}def __init__(self, num_classes=2, act=sign, sigmoid=False, softmax=False, scale=1, votes=1, bias=True):',
        f'{indent}{indent}super({name}, self).__init__()',
        f'{indent}{indent}self.votes = votes',
        f'{indent}{indent}self.num_classes = num_classes',
        f'{indent}{indent}if act == "sign":',
        f'{indent}{indent}{indent}self.act = torch.sign',
        f'{indent}{indent}elif act == "signb":',
        f'{indent}{indent}{indent}self.act = signb',
        f'{indent}{indent}elif act == "sigmoid":',
        f'{indent}{indent}{indent}self.act = torch.sigmoid_',
        f'{indent}{indent}elif act == "relu":',
        f'{indent}{indent}{indent}self.act = torch.relu_',
        '',
        f'{indent}{indent}if softmax:',
        f'{indent}{indent}{indent}if num_classes < 3:',
        f'{indent}{indent}{indent}{indent}' + 'raise ValueError("num_classes expect larger than 3, but got {num_classes}")',
        f'{indent}{indent}{indent}self.signb = softmax_',
        f'{indent}{indent}else:',
        f'{indent}{indent}{indent}self.signb = torch.sigmoid if sigmoid else signb',
        '',
    ]

    b = []
    names = []
    for i, layer in enumerate(structure):
        if 'fc' == layer.name:
            layer.layer_type = layer.layer_type.replace('Linear', 'Conv1d')

        main_body = f'self.{layer.name}{i + 1}_si = {layer.layer_type}'
        if layer.name == 'conv':
            parameters = [
                f'{layer.in_channel}' if i == 0 else f'{layer.in_channel} * votes',
                f'{layer.out_channel} * votes',
            ]
            if layer.kernel_size:
                parameters.append(f'kernel_size={layer.kernel_size}')
            if layer.padding:
                parameters.append(f'padding={layer.padding}')
            if layer.bias:
                parameters.append(f'bias=bias', )
        elif layer.name == 'fc':
            parameters = [
                '1' if i == 0 else 'votes',
                f'{layer.out_channel} * votes',
                f'kernel_size={layer.in_channel}',
            ]
            if layer.padding:
                parameters.append(f'padding={layer.padding}')
            if layer.bias:
                parameters.append(f'bias=bias', )
        else:
            raise ValueError('Something wrong with the code, trigger error')
        if i > 0:
            parameters.append('groups=votes')
        param = ", ".join(parameters)
        command = f'{indent}{indent}{main_body}({param})'
        b.append(command)
        names.append(f'"{layer.name}{i + 1}_si"')

    layer_list = ", ".join(names)
    b.append(f'{indent}{indent}self.layers = [{layer_list}]')
    b.append('')

    c = [
        f'{indent}def forward(self, out):',

    ]

    for i, layer in enumerate(structure):
        temp = []  # initial a temp list for saving each layer's forward command

        if layer.name == 'fc':
            if i == 0:
                temp += [f'{indent}{indent}out.unsqueeze_(dim=1)']
            else:
                temp += [f'{indent}{indent}out = out.reshape((out.size(0), self.votes, -1))']
        if i < len(structure) - 1:
            temp += [f'{indent}{indent}out = self.%s(out)' % names[i].replace('"','')]
            if layer.scale:
                if layer.kernel_size:
                    scale = math.sqrt(layer.out_channel) * layer.kernel_size
                else:
                    scale = math.sqrt(layer.out_channel)
                temp += [f'{indent}{indent}out = {layer.act}(out)'+' * {:.4f}'.format(1/scale)]
            else:
                temp += [f'{indent}{indent}out = {layer.act}(out)']
            # temp += [f'{indent}{indent}out = {layer.act}(out)']
            if layer.pool_size:
                if layer.pool_type == 'avg':
                    temp += [f'{indent}{indent}out = F.avg_pool2d(out, {layer.pool_size})']
                elif layer.pool_type == 'fs':
                    temp += [f'{indent}{indent}out = F.relu(out)']
                    temp += [
                        f'{indent}{indent}out = F.avg_pool2d(out, {layer.pool_size}) * {layer.pool_size ** 2}']

        else:
            temp += [
                f'{indent}{indent}out = self.%s(out)' % names[i].replace('"',''),
                f'{indent}{indent}out = out.reshape((out.size(0), self.votes, self.num_classes))',
                f'{indent}{indent}if self.num_classes == 1:',
                f'{indent}{indent}{indent}out = self.signb(out).squeeze(dim=-1)',
                f'{indent}{indent}{indent}out = out.mean(dim=1).round()',
                f'{indent}{indent}else:',
                f'{indent}{indent}{indent}out = self.signb(out)',
                f'{indent}{indent}{indent}out = out.mean(dim=1).argmax(dim=-1)',
                '',
                f'{indent}{indent}return out'

            ]



        c += temp

    content = a + b + c

    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    with open('../core/ensemble_model.py', 'r') as f:
        contents = f.readlines()

    id1 = contents.index('arch = {}\n')
    id2 = contents.index("if __name__ == '__main__':\n")
    part1 = contents[:id1]
    part2 = contents[id1:id2]
    part3 = contents[id2:]

    part1 += [line + '\n' for line in content]
    part1 += '\n\n'
    part2 += f"arch['{name.lower()}'] = {name}\n"

    new_contents = part1 + part2 + part3

    with open('../core/ensemble_model.py', 'w') as f:
        f.writelines(new_contents)
    # with open(os.path.join(save_path, f'{name}_ensemble.py'), 'w') as f:
    #     for line in contents:
    #         f.write(line + '\n')


def get_code_for_structure(name, structure, save_path):

    get_code_for_cnn01(name, structure, save_path)
    get_code_for_ensemble(name, structure, save_path)


if __name__ == '__main__':

    save_path = '../core/structures'


    name = 'toy3ssss100scale'
    structure = [

        Layer(layer_type='nn.Conv2d', in_channel=3,
              out_channel=16, kernel_size=3, padding=1, bias=True,
              pool_size=2, act='sign', pool_type='avg', scale=True),
        Layer(layer_type='nn.Conv2d', in_channel=16,
              out_channel=32, kernel_size=3, padding=1, bias=True,
              pool_size=2, act='sign', pool_type='avg', scale=True),
        Layer(layer_type='nn.Conv2d', in_channel=32,
              out_channel=64, kernel_size=3, padding=1, bias=True,
              pool_size=2, reshape=True, act='sign', pool_type='avg', scale=True),
        Layer(layer_type='nn.Linear', in_channel=64 * 4 * 4,
              out_channel=100, bias=True, act='sign', scale=True),
        Layer(layer_type='nn.Linear', in_channel=100,
              out_channel='num_classes', bias=True, act='relu'),
    ]

    get_code_for_structure(name, structure, save_path)