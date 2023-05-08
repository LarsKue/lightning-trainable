
import torch


def accuracy(logits, targets, *, k=1):
    """
    Computes the top-k accuracy for the specified values of k
    @param logits: Logits tensor of shape (batch_size, num_classes)
    @param targets: One-hot encoded target labels of shape (batch_size, num_classes)
    @param k: Number of top predictions to consider, e.g. k=1 for top-1 accuracy
    """
    if logits.dim() != 2 or logits.shape != targets.shape:
        raise ValueError(f"Expected logits and one-hot encoded targets to have shape (batch_size, num_classes), "
                         f"but got {logits.shape} and {targets.shape}.")

    batch_size, num_classes = logits.shape
    if not 1 <= k <= num_classes:
        raise ValueError(f"Expected k to be in the range [1, num_classes], but got {k=} for {num_classes=}.")

    # get top-k class indices
    logits, labels = torch.topk(logits, k, dim=1, largest=True)

    # compute correct predictions in shape (batch_size,)
    correct = torch.gather(targets, 1, labels)
    correct = torch.any(correct, dim=1)

    return correct.mean(dim=0, dtype=torch.float32)
