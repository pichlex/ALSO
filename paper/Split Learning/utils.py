import torch
import numpy as np


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_accuracy_at_k(outputs, labels, k=5):
    _, preds = torch.topk(outputs, k, dim=1, largest=True, sorted=True)
    correct = preds.eq(labels.view(-1, 1).expand_as(preds))
    return correct.sum().item() / labels.size(0)


