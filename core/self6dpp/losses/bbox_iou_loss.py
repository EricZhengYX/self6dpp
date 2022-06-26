import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.ops import box_iou


if hasattr(torchvision.ops, "box_area"):
    box_area = torchvision.ops.box_area
else:
    def box_area(boxes: Tensor) -> Tensor:
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.

        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format

        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


if hasattr(torchvision.ops, "generalized_box_iou"):
    generalized_box_iou = torchvision.ops.generalized_box_iou
else:
    def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """
        Return generalized intersection-over-union (Jaccard index) of boxes.

        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Arguments:
            boxes1 (Tensor[N, 4])
            boxes2 (Tensor[M, 4])

        Returns:
            generalized_iou (Tensor[N, M]): the NxM matrix containing the pairwise generalized_IoU values
            for every element in boxes1 and boxes2
        """

        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union

        lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        areai = whi[:, :, 0] * whi[:, :, 1]

        return iou - (areai - union) / areai


# Implementation adapted from torchvision.ops.boxes.box_iou
def box_iou_b11(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[N, 4])

    Returns:
        iou (Tensor[N]): the [N] vector containing the pairwise IoU values for every element-couple in boxes1 and boxes2
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape[0] == boxes2.shape[0]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter  # [N]

    iou = inter / union  # [N]
    return iou


# Implementation adapted from torchvision.ops.boxes.generalized_box_iou
def generalized_box_iou_b11(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[N, 4])

    Returns:
        generalized_iou (Tensor[N]): the [N] vector containing the pairwise generalized_IoU values
        for every element-couple in boxes1 and boxes2
    """

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape[0] == boxes2.shape[0]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter  # [N]

    iou = inter / union  # [N]

    lt_enclosing = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb_enclosing = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)  # [N,2]
    area_enclosing = wh_enclosing[:, 0] * wh_enclosing[:, 1]  # [N]

    return iou - (area_enclosing - union) / area_enclosing  # [N]


# Implementation adapted from torchvision.ops.boxes.masks_to_boxes (0.12.0)
def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        if mask.min() == mask.max() == 0:
            continue

        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


class IOULoss(nn.Module):
    def __init__(self, loss_type='iou', reduction='mean', result_mode='b11', esp=1e-5):
        super(IOULoss, self).__init__()
        if reduction.lower() == 'mean':
            self.reduction = 'mean'
        elif reduction.lower() == 'sum':
            self.reduction = 'sum'
        else:
            raise ValueError('An unknown reduction method is assigned in IOULoss!')

        if result_mode.lower() == 'b11':
            self.result_mode = 0
            if loss_type.lower() == 'iou':
                self.loss_func = box_iou_b11
                self.loss_func_flag = 'iou'
            elif loss_type.lower() == 'giou':
                self.loss_func = generalized_box_iou_b11
                self.loss_func_flag = 'giou'
            else:
                raise ValueError('An unknown loss function type is assigned in IOULoss!')
        elif result_mode.lower() == 'mn':
            self.result_mode = 1
            if loss_type.lower() == 'iou':
                self.loss_func = box_iou
                self.loss_func_flag = 'iou'
            elif loss_type.lower() == 'giou':
                self.loss_func = generalized_box_iou
                self.loss_func_flag = 'giou'
            else:
                raise ValueError('An unknown loss function type is assigned in IOULoss!')
        else:
            raise ValueError('Result mode in IOULoss should be ether \'b11\' or \'mn\' !')

        self.esp = esp

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        assert boxes1.ndim == 2 and boxes1.shape[1] == 4 and boxes2.ndim == 2 and boxes2.shape[1] == 4, \
            'Wrong input shape: {s1}, {s2}'.format(s1=str(boxes1.shape), s2=str(boxes2.shape))
        assert boxes1.device == boxes2.device, \
            'Inputs should be on the same device, but {d1} and {d2} are found'.format(d1=boxes1.device, d2=boxes2.device)

        if self.result_mode == 1:
            return self.loss_func(boxes1, boxes2)

        assert boxes1.shape[0] == boxes2.shape[0], 'With result_mode==\'b11\', inputs should both have shape b*4.'
        b = boxes1.shape[0]

        loss = self.loss_func(boxes1=boxes1, boxes2=boxes2)

        if self.loss_func_flag == 'giou':
            loss = (1 - loss).sum()
        else:  # iou
            loss = - torch.log(loss.clamp(min=self.esp)).sum()

        if self.reduction == 'mean':
            return loss / b
        else:
            return loss
