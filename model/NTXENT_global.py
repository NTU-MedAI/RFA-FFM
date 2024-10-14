import torch
import numpy as np
from model.Hier_Net import get_device

torch.multiprocessing.set_sharing_strategy('file_system')


torch.cuda.empty_cache()
pos = []
enhance1 = []
enhance2 = []
m_lis = []


class NTXentLoss(torch.nn.Module):
    def __init__(self):
        super(NTXentLoss, self).__init__()
        self.temperature = 0.1
        self.device = get_device()
        self.lambda_1 = 0.5
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_function = torch.nn.CosineSimilarity(dim=-1)

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        device = get_device()
        return mask.to(device)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        representations = torch.cat([x2, x1], dim=0)

        similarity_matrix = self.similarity_function(representations.unsqueeze(1), representations.unsqueeze(0))

        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)