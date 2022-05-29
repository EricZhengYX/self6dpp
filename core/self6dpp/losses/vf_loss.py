import torch
from torch import nn
import torch.nn.functional as F


class VFLoss(nn.Module):
    def __init__(self, with_l1=True, with_cs=True):
        super().__init__()
        assert with_l1 or with_cs
        if with_l1:
            self._with_l1 = True
            self._l1_loss_func = nn.L1Loss(reduction="mean")
        else:
            self._with_l1 = False
            self._l1_loss_func = None
        if with_cs:
            self._with_cs = True
        else:
            self._with_cs = False

    def forward(self, out_vf: torch.Tensor, gt_vf: torch.Tensor, mask: torch.Tensor):
        masked_out_vf = mask * out_vf
        masked_gt_vf = mask * gt_vf

        loss = 0.0
        if self._with_l1:
            loss += self._l1_loss_func(masked_out_vf, masked_gt_vf)
        if self._with_cs:
            loss += self._cosine_similarity_loss_vf(masked_out_vf, masked_gt_vf, mask)
        return loss

    def _cosine_similarity_loss_vf(self, out_vf: torch.Tensor, gt_vf: torch.Tensor, mask: torch.Tensor):
        b, c, _, w, h = out_vf.shape

        num_foreground_pix = torch.count_nonzero(mask.detach()).item() * c
        cs_vf = F.cosine_similarity(out_vf, gt_vf, dim=2)  # b, c, w, h

        minus_cs_vf = torch.ones_like(cs_vf) - cs_vf

        masked_minus_cs_vf = mask.squeeze(2) * minus_cs_vf

        return masked_minus_cs_vf.sum() / num_foreground_pix
