import warnings

import torch
import torch.nn as nn


# Implementation adapted from torchvision.ops.boxes.box_iou
def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        mask1 (Tensor[B, N, M])
        mask2 (Tensor[B, N, M])

    Returns:
        iou (Tensor[B]): the [B] vector containing the pairwise Mask-IoU values for every element-couple in mask1 and mask2
    """
    assert mask1.shape == mask2.shape, \
        'Mask1 and Mask2 should have same shape, but got {s1} != {s2}'.format(s1=mask1.shape, s2=mask2.shape)
    b = mask1.shape[0]
    area1 = torch.sum(mask1.view(b, -1), dim=1)  # [B]
    area2 = torch.sum(mask2.view(b, -1), dim=1)  # [B]

    inter = torch.mul(mask1, mask2)  # [B, N, M]
    inter_size = inter.view(b, -1).sum(dim=1)  # [B]

    union_size = area1 + area2 - inter_size  # [B]

    iou = inter_size / union_size  # [B]
    return iou


class MaskIOULoss(nn.Module):
    def __init__(self, loss_type='iou', reduction='mean', result_mode='b11', esp=1e-7, loss_weight=1):
        super(MaskIOULoss, self).__init__()
        if reduction.lower() == 'mean':
            self.reduction = 'mean'
        elif reduction.lower() == 'sum':
            self.reduction = 'sum'
        else:
            raise ValueError('An unknown reduction method is assigned in IOULoss!')

        self.loss_func = mask_iou
        self.loss_func_flag = 'iou'
        self.loss_weight = loss_weight
        # if result_mode.lower() == 'b11':
        #     self.result_mode = 0
        #     if loss_type.lower() == 'iou':
        #         self.loss_func = box_iou_b11
        #         self.loss_func_flag = 'iou'
        #     elif loss_type.lower() == 'giou':
        #         self.loss_func = generalized_box_iou_b11
        #         self.loss_func_flag = 'giou'
        #     else:
        #         raise ValueError('An unknown loss function type is assigned in IOULoss!')
        # elif result_mode.lower() == 'mn':
        #     self.result_mode = 1
        #     if loss_type.lower() == 'iou':
        #         self.loss_func = box_iou
        #         self.loss_func_flag = 'iou'
        #     elif loss_type.lower() == 'giou':
        #         self.loss_func = generalized_box_iou
        #         self.loss_func_flag = 'giou'
        #     else:
        #         raise ValueError('An unknown loss function type is assigned in IOULoss!')
        # else:
        #     raise ValueError('Result mode in IOULoss should be ether \'b11\' or \'mn\' !')
        self.esp = esp

    def forward(self, mask1: torch.Tensor, mask2: torch.Tensor):
        assert mask1.ndim == 3 and mask2.ndim == 3 and mask1.shape[1:] == mask2.shape[1:], \
            'Wrong input shape: {s1}, {s2}'.format(s1=str(mask1.shape), s2=str(mask2.shape))
        assert mask1.device == mask2.device, \
            'Inputs should be on the same device, but {d1} and {d2} are found'.format(d1=mask1.device, d2=mask2.device)

        assert mask1.shape[0] == mask2.shape[0], \
            'With result_mode==\'b11\', inputs should both have same batchsize. But got {s1} != {s2}'\
                .format(s1=mask1.shape[0], s2=mask2.shape[0])
        valid_entries = mask1.shape[0]

        loss: torch.Tensor = self.loss_func(mask1=mask1, mask2=mask2)

        if self.loss_func_flag == 'giou':
            # loss = (1 - loss).sum()
            raise ValueError
        else:  # iou
            valid_entries = loss > self.esp
            valid_loss = loss[valid_entries]
            valid_loss = - torch.log(valid_loss)

        num_valid_entries = len(valid_loss)
        if num_valid_entries == 0:
            # warnings.warn('Mask IOU is failure in a whole batch')
            return torch.tensor(0.0, device=mask1.device), valid_entries.int().tolist()

        if self.reduction == 'mean':
            return valid_loss.sum() / num_valid_entries * self.loss_weight, valid_entries.int().tolist()
        else:
            return valid_loss.sum() * self.loss_weight, valid_entries.int().tolist()
