import torch
from torch import nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self, out0, out1):
        device = out0.device
        batch_size = out0.size(0)
        print('batch_size: ', batch_size)

        out0 = F.normalize(out0, dim=0)
        out1 = F.normalize(out1, dim=0)

        output = torch.cat([out0, out1], 0)

        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature

        logits = logits[~torch.eye(2 * batch_size, dtype=torch.bool, device=out0.device)].view(2 * batch_size, -1)
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels])

        loss = self.cross_entropy(logits, labels)

        return loss


class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, out0, out1):
        loss = self.mse(out0, out1)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        target = target.float()
        pt = torch.softmax(inputs, dim=1)
        p = pt[:, 1]
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p))
        return loss.mean()


class JointLoss(nn.Module):
    def __init__(self, loss, cl_loss, alpha=0.5):
        super(JointLoss, self).__init__()
        self.loss = loss
        self.cl_loss = cl_loss
        self.alpha = alpha

    def forward(self, output, target, vec0=None, vec1=None):
        if self.cl_loss is None:
            loss = self.loss(output, target)
        else:
            loss = self.alpha * self.cl_loss(vec0, vec1) + (1 - self.alpha) * self.loss(output, target)

        return loss


def build_loss(cfg, weight):
    if cfg.DATA.TASK_TYPE == 'classification':
        if weight is not None:
            loss = nn.CrossEntropyLoss() if not cfg.LOSS.FL_LOSS else FocalLoss(alpha=1/weight[0])
        else:
            loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()
    cl_loss = NTXentLoss(temperature=cfg.LOSS.TEMPERATURE) if cfg.LOSS.CL_LOSS else None

    joint_loss = JointLoss(loss=loss,
                           cl_loss=cl_loss,
                           alpha=cfg.LOSS.ALPHA)

    return joint_loss
