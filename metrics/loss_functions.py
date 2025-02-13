from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from deepsat.utils.config_files_utils import get_params_values
from copy import deepcopy


def get_loss(config, device, reduction='mean'):
    model_config = config['MODEL']
    loss_config = config['SOLVER']

    print(loss_config['loss_function'])

    if type(loss_config['loss_function']) in [list, tuple]:
        loss_fun = []
        loss_types = deepcopy(loss_config['loss_function'])
        config_ = deepcopy(config)
        for loss_fun_type in loss_types:
            config_['SOLVER']['loss_function'] = loss_fun_type
            loss_fun.append(get_loss(config_, device, reduction=reduction))
        return loss_fun

    # Contrastive Loss -----------------------------------------------------------------------
    if loss_config['loss_function'] in ['contrastive_loss', 'masked_contrastive_loss']:
        pos_weight = get_params_values(config['SOLVER'], 'pos_weight', 1.0)
        print("cscl positive weight: ", pos_weight)
        return MaskedContrastiveLoss(pos_weight=pos_weight, reduction=reduction)

    # Binary Cross-Entropy Loss -----------------------------------------------------------
    if loss_config['loss_function'] == 'binary_cross_entropy':
        if reduction is None:
            reduction = 'none'
        return nn.BCEWithLogitsLoss(reduction=reduction)

    if loss_config['loss_function'] == 'masked_binary_cross_entropy':
        pos_weight = config['SOLVER']['pos_weight']
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        return MaskedBinaryCrossEntropy(reduction=reduction, pos_weight=pos_weight)

    # Cross-Entropy Loss ------------------------------------------------------------------
    elif loss_config['loss_function'] == 'cross_entropy':
        num_classes = get_params_values(model_config, 'num_classes', None)
        weight = torch.Tensor(num_classes * [1.0]).to(device)
        if loss_config['class_weights'] not in [None, {}]:
            for key in loss_config['class_weights']:
                weight[key] = loss_config['class_weights'][key]
        return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    # Weighted Cross-Entropy Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'weight_cross_entropy':
        label_smoothing = get_params_values(loss_config, "label_smoothing", 0.0)
        pos_weight = loss_config['pos_weight']
        if pos_weight is not None:
            weight_1 = 1. / pos_weight
            weight_2 = 1.
            total_weight = weight_1 + weight_2
            weight_1 = weight_1 / total_weight
            weight_2 = weight_2 / total_weight
            weight = torch.tensor([weight_1, weight_2], dtype=torch.float32).to(device)
            return torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

        else:
            return torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    # Masked Cross-Entropy Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_cross_entropy':
        mean = reduction == 'mean'
        return MaskedCrossEntropyLoss(mean=mean)

    # Focal Loss --------------------------------------------------------------------------
    elif loss_config['loss_function'] in ['focal_loss', 'masked_focal_loss']:
        gamma = get_params_values(loss_config, "gamma", 1.0)
        try:
            alpha = get_params_values(loss_config, "alpha", None)
        except ValueError:
            alpha = None
        if loss_config['loss_function'] == 'focal_loss':
            return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        elif loss_config['loss_function'] == 'masked_focal_loss':
            return MaskedFocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    # Masked Multiclass Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'masked_dice_loss':
        label_smoothing = get_params_values(loss_config, "label_smoothing", 0.0)
        return MaskedDiceLoss(reduction=reduction, device=device, label_smoothing=label_smoothing)

    # Tversky Loss -----------------------------------------------------------
    elif loss_config['loss_function'] in ['tversky_loss', 'focal_tversky_loss']:
        smooth = get_params_values(loss_config, "smooth", 1)
        alpha = get_params_values(loss_config, "alpha", 0.5)
        beta = get_params_values(loss_config, "beta", 0.5)
        gamma = 1.0 if loss_config['loss_function'] == 'tversky_loss' else get_params_values(loss_config, "gamma", 1.0)
        return FocalTverskyLoss(smooth=smooth, alpha=alpha, beta=beta, gamma=gamma, reduction=reduction)

    # Combined Loss -----------------------------------------------------------
    elif loss_config['loss_function'] == 'combined_dice_ce':
        loss_fns = [MaskedDiceLoss(reduction=reduction, device=device)]
        label_smoothing = get_params_values(loss_config, "label_smoothing", 0.0)
        pos_weight = loss_config['pos_weight']
        if pos_weight is not None:
            weight_1 = 1. / pos_weight
            weight_2 = 1.
            total_weight = weight_1 + weight_2
            weight_1 = weight_1 / total_weight
            weight_2 = weight_2 / total_weight
            weight = torch.tensor([weight_1, weight_2], dtype=torch.float32).to(device)
            loss_fns.append(torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing))

        else:
            loss_fns.append(torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing))

        weights = get_params_values(loss_config, "loss_weights", [0.5, 0.5])
        return CombinedLoss(loss_fns, weights)


def per_class_loss(criterion, logits, labels, unk_masks, n_classes):
    class_loss = []
    class_counts = []
    for class_ in range(n_classes):
        idx = labels == class_
        class_loss.append(
            criterion(logits[idx.repeat(1, 1, 1, n_classes)].reshape(-1, n_classes),  # ???
                      labels[idx].reshape(-1, 1),
                      unk_masks[idx].reshape(-1, 1)).detach().cpu().numpy()
        )
        class_counts.append(unk_masks[idx].sum().cpu().numpy())
    class_loss = np.array(class_loss)
    class_counts = np.array(class_counts)
    return np.nan_to_num(class_loss, nan=0.0), class_counts


class MaskedContrastiveLoss(torch.nn.Module):
    def __init__(self, pos_weight=1, reduction="mean"):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedContrastiveLoss, self).__init__()

        self.pos_weight = pos_weight
        self.reduction = reduction
        self.h = 1e-7

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        else:
            target = ground_truth[0]
            mask = ground_truth[1].to(torch.float32)

        loss = - self.pos_weight * target * logits + (1 - target) * logits
        if mask is not None:
            loss = mask * loss

        if self.reduction == "mean":
            return loss.mean()  # loss.sum() / (mask.sum() - 1)
        return loss


class MaskedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction="mean", pos_weight=None):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedBinaryCrossEntropy, self).__init__()
        self.reduction = reduction
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits, ground_truth):
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
        elif len(ground_truth) == 1:
            target = ground_truth[0]
        else:
            target = ground_truth[0][ground_truth[1]]
            logits = logits[ground_truth[1]]
        return self.loss_fn(logits, target)


class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, mean=True):
        """
        mean: return mean loss vs per element loss
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.mean = mean

    def forward(self, logits, ground_truth):
        """
            Args:
                logits: (N,T,H,W,...,NumClasses)A Variable containing a FloatTensor of size
                    (batch, max_len, num_classes) which contains the
                    unnormalized probability for each class.
                target: A Variable containing a LongTensor of size
                    (batch, max_len) which contains the index of the true
                    class for each corresponding step.
                length: A Variable containing a LongTensor of size (batch,)
                    which contains the length of each data in a batch.
            Returns:
                loss: An average loss value masked by the length.
            """
        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        if mask is not None:
            mask_flat = mask.reshape(-1, 1)  # (N*H*W x 1)
            nclasses = logits.shape[-1]
            logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_logits_flat = logits_flat[mask_flat.repeat(1, nclasses)].view(-1, nclasses)
            target_flat = target.reshape(-1, 1)  # (N*H*W x 1)
            masked_target_flat = target_flat[mask_flat].unsqueeze(dim=-1).to(torch.int64)
        else:
            masked_logits_flat = logits.reshape(-1, logits.size(-1))  # (N*H*W x Nclasses)
            masked_target_flat = target.reshape(-1, 1).to(torch.int64)  # (N*H*W x 1)
        masked_log_probs_flat = torch.nn.functional.log_softmax(masked_logits_flat)  # (N*H*W x Nclasses)
        masked_losses_flat = -torch.gather(masked_log_probs_flat, dim=1, index=masked_target_flat)  # (N*H*W x 1)
        if self.mean:
            return masked_losses_flat.mean()
        return masked_losses_flat


class MaskedFocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(MaskedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1, 1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(-1, target.unsqueeze(-1))
        logpt = logpt.reshape(-1)
        pt = logpt.exp()  # Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")

# Adapted for Binary Classification
class MaskedDiceLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """

    def __init__(self, reduction=None, label_smoothing=0., device='cuda'):
        super(MaskedDiceLoss, self).__init__()
        self.reduction = reduction
        self.device = device
        self.label_smoothing = label_smoothing

    def forward(self, logits, ground_truth):

        if type(ground_truth) == torch.Tensor:
            target = ground_truth
            mask = None
        elif len(ground_truth) == 1:
            target = ground_truth[0]
            mask = None
        elif len(ground_truth) == 2:
            target, mask = ground_truth
        else:
            raise ValueError("ground_truth parameter for MaskedCrossEntropyLoss is either (target, mask) or (target)")

        target = target.reshape(-1).to(torch.int64)
        logits = logits.reshape(-1, logits.shape[-1])

        if mask is not None:
            mask = mask.reshape(-1, 1)
            target = target[mask]
            logits = logits[mask.repeat(1, logits.shape[-1])].reshape(-1, logits.shape[-1])

        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = logits.shape[-1]
            smoothing_value = self.label_smoothing / (num_classes - 1)
            target_onehot = torch.full((target.size(0), num_classes), smoothing_value, device=self.device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            # Standard one-hot encoding if no label smoothing
            target_onehot = F.one_hot(target, num_classes=logits.shape[-1]).to(torch.float32)

        predicted_prob = F.softmax(logits, dim=-1)
        predicted_prob_pos = predicted_prob[:, 1]
        target_pos = target_onehot[:, 1]

        inter = (predicted_prob_pos * target_pos).sum()
        union = predicted_prob_pos.sum() + target_pos.sum()

        loss = 1 - 2 * inter / union

        # Not used but kept for future reference
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "DiceLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class FocalLoss(nn.Module):
    """
    Credits to  github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, alpha=None, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "FocalLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


# Tversky Loss adapted from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
# Specific to binary classification
class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.5, beta=0.5, gamma=1., reduction=None): # Default to DICE
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        #comment out if your model contains a sigmoid or equivalent activation layer
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C

        inputs = F.softmax(inputs, dim=1)

        #flatten label and prediction tensors
        targets = targets.view(-1)

        #True Positives, False Positives & False Negatives in Binary case
        TP = (inputs[:, 1] * targets).sum()
        FP = ((1-targets) * inputs[:, 1]).sum()
        FN = (targets * inputs[:, 0]).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        loss = (1 - Tversky)**self.gamma

        # Not used but kept for future reference
        if self.reduction is None:
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(
                "TverskyLoss: reduction parameter not in list of acceptable values [\"mean\", \"sum\", None]")


class CombinedLoss(nn.Module):
    def __init__(self, loss_fns: List[nn.Module], weights: List[float]):
        super(CombinedLoss, self).__init__()
        self.loss_fns = loss_fns
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0
        for i, loss_fn in enumerate(self.loss_fns):
            loss += self.weights[i] * loss_fn(inputs, targets)
        return loss
