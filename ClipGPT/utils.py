import torch

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support


label_freq = [2327, 10915, 9761, 14976, 27790, 32431, 13589, 18789, 26274, 4621, 3040, 1731, 754, 943, 871,
              25891, 66152, 58515, 18254,
              32370, 913, 2367, 1298, 12566, 609, 426, 751, 1667, 1630, 738, 560, 404, 8774, 1650, 186, 283, 405,
              29508, 41985, 12913,
              2416, 6837, 36580, 7853, 14125,
              2706, 9759, 1764, 16064, 12979, 21599, 7493, 42045, 3436, 2295, 734, 718, 405, 54]


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update model's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)


def save_clf_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, Acc, final_args):
    """
    Saves model checkpoint.
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'Acc': Acc,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def calc_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc


def calc_classwise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
    return classwise_acc


def calc_map(y_true, y_scores):
    mAP = average_precision_score(y_true, y_scores,average=None)
    return mAP


def calc_precision_recall_fscore(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division = 1)
    return(precision, recall, fscore)
