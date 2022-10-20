import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import BCELoss as bceloss
from torch.nn import NLLLoss as nllloss
from torch.nn import CrossEntropyLoss as celoss

class ZeroOneLoss(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, kind='imbalanced', num_classes=2):
        super(ZeroOneLoss, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs_, target):

        inputs = F.relu(inputs_.sign())
        while len(target.size()) < len(inputs.size()) - 1:
            target = target.unsqueeze(dim=1)
        inputs = inputs.squeeze(dim=-1)

        if self.kind == 'imbalanced':
            imbalanced_acc = inputs.eq(target).float().mean(dim=0)
            loss = 1 - imbalanced_acc
            return loss

        elif self.kind == 'balanced':
            positive = target.sum()
            negative = target.size(0) - positive
            positive_correct = ((inputs + target) == 2).char()
            negative_correct = ((inputs + target) == 0).char()
            balanced_acc = 0.5 * (positive_correct.sum(dim=0).float() / positive +
                                  negative_correct.sum(dim=0).float() / negative)

            loss = 1 - balanced_acc

            return loss

        elif self.kind == 'combined':
            positive = target.sum()
            negative = target.size(0) - positive
            positive_correct = ((inputs + target) == 2).char()
            negative_correct = ((inputs + target) == 0).char()
            balanced_acc = 0.5 * (positive_correct.sum(dim=0).float() / positive +
                                  negative_correct.sum(dim=0).float() / negative)
            imbalanced_acc = inputs.eq(target).float().mean(dim=0)

            loss = 2 - imbalanced_acc - balanced_acc

            return loss

        else:
            raise ValueError(f'{self.kind} is not available')


class BCELoss(nn.Module):
    """
    Multi dimension BCE loss
    """

    def __init__(self, ):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        return

    # def _bce(self, x):
    #
    #     return F.binary_cross_entropy_with_logits(x, reduction='none')

    def forward(self, inputs, target):
        """

        :param inputs: nrows * -1
        :param target: nrows * 1
        :return:
        """
        # loss_group = criterion(yp, torch.stack([torch.stack([target for i in range(yp.size(1))], dim=1) for j in range(yp.size(2))],dim=2).type_as(yp).unsqueeze(dim=-1))
        # loss_group = loss_group.mean(dim=0, keepdim=True).squeeze()

        # loss_group = criterion(yp, torch.stack([target for i in range(yp.size(1))], dim=1).type_as(yp).unsqueeze(dim=-1))
        # loss_group = loss_group.mean(dim=0).squeeze()
        # bce = bceloss(reduction='none')

        if len(inputs.size()) == 4:
            loss = self.bce(inputs, torch.stack(
                [torch.stack(
                    [target for i in range(inputs.size(1))], dim=1) for j in range(inputs.size(2))],
                 dim=2).type_as(inputs).unsqueeze(dim=-1))
        elif len(inputs.size()) == 3:
            loss = self.bce(inputs,
                             torch.stack([target for i in range(inputs.size(1))],
                                         dim=1).type_as(inputs).unsqueeze(dim=-1))
        elif len(inputs.size()) == 2:
            loss = self.bce(inputs, target.type_as(inputs).unsqueeze(dim=-1))

        return loss.mean(dim=0).squeeze(dim=-1)


class CrossEntropyLoss(nn.Module):
    """
    Multi dimension BCE loss
    """

    def __init__(self, num_classes=10):
        super(CrossEntropyLoss, self).__init__()
        self.ce = celoss(reduction='none')
        return

    def forward(self, inputs, target):
        """

        :param inputs: nrows * n_w * n_b * n_classes
        :param target: nrows * 1
        :return:
        """

        # ce = nllloss(reduction='none')
        # ce = celoss(reduction='none')
        if len(inputs.size()) == 4:
            loss = self.ce(inputs.permute(0, 3, 1, 2), torch.stack(
                [torch.stack(
                    [target for i in range(inputs.size(1))], dim=1) for j in range(inputs.size(2))],
                 dim=2))
        elif len(inputs.size()) == 3:
            loss = self.ce(inputs.permute(0, 2, 1), torch.stack([target for i in range(inputs.size(1))], dim=1))
        elif len(inputs.size()) == 2:
            loss = self.ce(inputs, target)
        return loss.mean(dim=0)

class ZeroOneLossMC(nn.Module):
    """

    :param inputs: nrows * n_w * n_b * n_classes
    :param target: nrows * 1
    :return:
    """

    def __init__(self, kind='balanced', num_classes=10):
        super(ZeroOneLossMC, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs, target):

        if len(inputs.size()) == 4:
            loss = torch.eq(inputs.argmax(dim=-1), torch.stack(
                [torch.stack(
                    [target for i in range(inputs.size(1))], dim=1) for j in range(inputs.size(2))],
                 dim=2))
        elif len(inputs.size()) == 3:
            loss = torch.eq(inputs.argmax(dim=-1), torch.stack([target for i in range(inputs.size(1))], dim=1))
        elif len(inputs.size()) == 2:
            loss = torch.eq(inputs.argmax(dim=-1), target)
        return 1 - loss.float().mean(dim=0)




class L1MC(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, kind='balanced', num_classes=10):
        super(L1MC, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs_, target):
        # inputs.float()
        inputs = F.relu(inputs_.sign())
        inputs_onehot = inputs
        if len(inputs_onehot.size()) == 4:
            inputs_onehot = inputs_onehot.permute(0, 3, 1, 2)
        if len(inputs_onehot.size()) == 3:
            inputs_onehot = inputs_onehot.permute(0, 2, 1)
        target_onehot = F.one_hot(target, num_classes=self.n_classes)
        while len(target_onehot.size()) < len(inputs_onehot.size()):
            target_onehot.unsqueeze_(dim=-1)
        # match = (inputs_onehot + target_onehot) // 2
        # loss = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()
        mis_match = (inputs_onehot - target_onehot).abs().float()
        loss = mis_match.mean(dim=1).mean(dim=0)
        return loss


class ZeroOneLossModified(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, kind='balanced', num_classes=10):
        super(ZeroOneLossModified, self).__init__()
        self.kind = kind
        self.n_classes = num_classes

        return

    def forward(self, inputs_, target):
        # inputs.float()
        inputs = F.relu(inputs_.sign())
        inputs_onehot = inputs
        if len(inputs_onehot.size()) == 4:
            inputs_onehot = inputs_onehot.permute(0, 3, 1, 2)
        if len(inputs_onehot.size()) == 3:
            inputs_onehot = inputs_onehot.permute(0, 2, 1)

        target_onehot = F.one_hot(target, num_classes=self.n_classes)

        while len(target_onehot.size()) < len(inputs_onehot.size()):
            target_onehot.unsqueeze_(dim=-1)
        target_weights = torch.zeros_like(target_onehot).float().fill_(0.1)
        target_weights[torch.arange(target.size(0)), target] = 0.9
        # match = (inputs_onehot + target_onehot) // 2
        # loss = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()
        mis_match = (inputs_onehot - target_onehot).abs() * target_weights
        loss = mis_match.sum(dim=1).mean(dim=0)
        return loss


class HingeLoss(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = self.c - target * inputs
        loss[loss < 0] = 0
        loss = loss.mean()



        return loss

class ConditionalEntropy(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(ConditionalEntropy, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        unique_target, target_counts = torch.unique(target, return_counts=True)
        target_probs = target_counts.float() / target.size(0)
        sum = 0
        for i, unique_value in enumerate(unique_target):
            sequence = inputs.flatten()[target == unique_value]
            sequence_prob = torch.unique(sequence, return_counts=True)[1].float() / sequence.size(0)
            sum += target_probs[i] * Categorical(probs=sequence_prob).entropy()



        return sum

class HingeLoss3(nn.Module):
    """Hinge-Loss

    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss3, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = self.c - (target * inputs).abs()
        loss[loss < 0] = 0
        loss1 = loss.mean()

        inputs_ = inputs.sign()
        loss2 = 1 - inputs_.eq(target).float().mean()
        return loss1


class HingeLoss2(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1.0):
        super(HingeLoss2, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c

        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = 2.0 * target - 1
        loss = -1.0 * target * inputs
        # loss[loss < 0] = 0
        loss = loss.mean()
        return loss

class CombinedLoss(nn.Module):
    """Zero-One-Loss
    Balanced or non-balanced
    Version: V1.0
    """

    def __init__(self, balance=True, num_classes=2, c=1):
        super(CombinedLoss, self).__init__()
        self.balance = balance
        self.n_classes = num_classes
        self.c = c
        return

    def forward(self, inputs, target):
        # inputs.float()
        target = target.reshape((-1, 1))
        target = (2.0 * target - 1)
        loss = 1 - target * inputs
        loss[loss < 0] = 0
        loss  = loss
        loss_h = loss.mean()

        inputs = (torch.sign(inputs) + 1 ) // 2
        inputs_onehot = F.one_hot(inputs.flatten().long(), num_classes=self.n_classes)
        target_onehot = F.one_hot(target, num_classes=self.n_classes)
        match = (inputs_onehot + target_onehot) // 2
        loss_01 = 1 - (match.sum(dim=0) / target_onehot.sum(dim=0).float()).mean()

        return loss_h + loss_01


criterion = {}
criterion['mce'] = CrossEntropyLoss
criterion['01loss'] = ZeroOneLoss
criterion['bce'] = BCELoss
criterion['01lossmc'] = ZeroOneLossMC
criterion['l1mc'] = L1MC

if __name__ == '__main__':
    import torch.nn as nn

    m = F.softmax
    loss = nn.CrossEntropyLoss()
    input = torch.randn(size=(100, 3), requires_grad=True)
    import torch
    target = torch.empty(size=(100,)).random_(3).long()
    output = loss(m(input), target)
    print(output)
    loss = CrossEntropyLoss()
    output = loss(m(input), target.unsqueeze(dim=1))
    print(output)

