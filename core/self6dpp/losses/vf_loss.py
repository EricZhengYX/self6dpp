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
        """

        @param out_vf: b*#fps*2*64*64
        @param gt_vf: b*#fps*2*64*64
        @param mask: b*1*64*64
        @return: l1+cos loss
        """
        _mask = mask.unsqueeze(1)
        masked_out_vf = _mask * out_vf
        masked_gt_vf = _mask * gt_vf

        loss = 0.0
        if self._with_l1:
            loss += self._l1_loss_func(masked_out_vf, masked_gt_vf)
        if self._with_cs:
            loss += self._cosine_similarity_loss_vf(masked_out_vf, masked_gt_vf, _mask)
        return loss

    def _cosine_similarity_loss_vf(self, out_vf: torch.Tensor, gt_vf: torch.Tensor, mask: torch.Tensor):
        b, c, _, w, h = out_vf.shape
        assert out_vf.shape == gt_vf.shape, "{} != {}".format(out_vf.shape, gt_vf.shape)
        assert mask.shape == (b, 1, 1, w, h), mask.shape

        # num_foreground_pix = torch.count_nonzero(mask.detach()).item() * c
        num_foreground_pix = (mask != 0).sum().item() * c
        cs_vf = F.cosine_similarity(out_vf, gt_vf, dim=2)  # b, c, w, h

        minus_cs_vf = torch.ones_like(cs_vf) - cs_vf

        masked_minus_cs_vf = mask.squeeze(2) * minus_cs_vf

        return masked_minus_cs_vf.sum() / num_foreground_pix
