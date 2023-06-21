import torch
from typing import Optional

def compute_confusion_matrix(logits: torch.Tensor,
                             targets: torch.Tensor,
                             activation_function: Optional[callable] = None,
                             threshold: float = 0.5) -> tuple[int, int, int, int]:

    if not torch.is_tensor(logits) or len(logits.shape) != 1 or not torch.is_floating_point(logits):
        raise TypeError("logits must be a 1D tensor of floating point data type")

    if not torch.is_tensor(targets) or len(targets.shape) != 1 or \
            (targets.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise TypeError("targets must be a 1D tensor of boolean or integer data type")

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same length")

    if activation_function is not None:
        logits = activation_function(logits)

    # here threshold operation
    predicted_labels = torch.where(logits >= threshold, torch.tensor(1), torch.tensor(0))

    # confusion matrix entries (true positives, false negatives, false positives, true negatives)
    true_positives = torch.sum(predicted_labels * targets).item()
    false_negatives = torch.sum((1 - predicted_labels) * targets).item()
    false_positives = torch.sum(predicted_labels * (1 - targets)).item()
    true_negatives = torch.sum((1 - predicted_labels) * (1 - targets)).item()

    return true_positives, false_negatives, false_positives, true_negatives

if __name__ == "__main__":
    torch.manual_seed(123)
    logits = torch.rand(size=(10,)) * 10 - 5
    targets = torch.randint(low=0, high=2, size=(10,))
    tp, fn, fp, tn = compute_confusion_matrix(
    logits, targets, activation_function=torch.sigmoid)
    print(logits)
    print(targets)
    print(f"{tp=}, {fn=}, {fp=}, {tn=}")