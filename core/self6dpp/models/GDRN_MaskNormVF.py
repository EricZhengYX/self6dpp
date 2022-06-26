import copy
import logging
from functools import partial
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.gdrn_modeling.datasets.lm_dataset_d2 import LM_13_OBJECTS, LM_OCC_OBJECTS
from core.utils.solver_utils import build_optimizer_with_params
from core.utils.data_utils import compute_vf_torch
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint

from core.self6dpp.losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from core.self6dpp.losses.l2_loss import L2Loss
from core.self6dpp.losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from core.self6dpp.losses.pm_loss import PyPMLoss
from core.self6dpp.losses.rot_loss import angular_distance, rot_l2_loss
from ..losses.vf_norm_loss import VFLoss, NORMLoss
from .model_utils import (
    compute_mean_re_te,
    get_neck,
    get_geo_head,
    get_mask_prob,
    get_pnp_net,
    get_rot_mat,
    get_xyz_doublemask_region_out_dim,
    get_xyz_mask_norm_vf_region_out_dim,
)
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .net_factory import BACKBONES
from lib.dr_utils.dib_renderer_x.renderer_dibr import Renderer_dibr
from core.utils.zoom_utils import batch_crop_resize
from core.utils.my_checkpoint import load_timm_pretrained

logger = logging.getLogger(__name__)


class GDRN_MaskNormVF(nn.Module):
    def __init__(self, cfg, backbone, geo_head_net, renderer: Renderer_dibr, render_models, neck=None, pnp_net=None):
        super().__init__()
        assert (
            cfg.MODEL.POSE_NET.NAME == "GDRN_MaskNormVF"
        ), cfg.MODEL.POSE_NET.NAME
        self.backbone = backbone
        self.neck = neck

        self.geo_head_net = geo_head_net
        self.pnp_net = pnp_net

        self.cfg = cfg
        (
            self.xyz_out_dim,
            self.mask_out_dim,
            self.region_out_dim,
            self.vf_out_dim,
            self.norm_out_dim,
        ) = get_xyz_mask_norm_vf_region_out_dim(cfg)

        # render
        self._renderer = renderer
        self._render_models = render_models

        # loss functions
        loss_cfg = cfg.MODEL.POSE_NET.LOSS_CFG
        self.ce_heatmap_loss_func = CrossEntropyHeatmapLoss(
            reduction="sum", weight=None
        )
        self.pm_loss_func = PyPMLoss(
            loss_type=loss_cfg.PM_LOSS_TYPE,
            beta=loss_cfg.PM_SMOOTH_L1_BETA,
            reduction="mean",
            loss_weight=loss_cfg.PM_LW,
            norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
            symmetric=loss_cfg.PM_LOSS_SYM,
            disentangle_t=loss_cfg.PM_DISENTANGLE_T,
            disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
            t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
            r_only=loss_cfg.PM_R_ONLY,
        )
        self.vf_loss_func = VFLoss()
        self.norm_loss_func = NORMLoss()

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        # yapf: disable
        if cfg.MODEL.POSE_NET.USE_MTL:
            self.loss_names = [
                "mask", "coor_x", "coor_y", "coor_z", "coor_x_bin", "coor_y_bin", "coor_z_bin", "region",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )
        # yapf: enable

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters."""
        super(GDRN_MaskNormVF, self).train(mode)
        cfg = self.cfg
        if cfg.MODEL.FREEZE_BN:
            logger.info("GDRN_MaskNormVF turns into training mode, but freeze bn...")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()  # do not update running stats

                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        return self

    def forward(
        self,
        x,
        gt_ego_rot=None,
        gt_trans=None,
        roi_coord_2d=None,
        roi_coord_2d_rel=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        forward_mode="geo",
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        pnp_net_cfg = net_cfg.PNP_NET

        device = x.device
        bs = x.shape[0]
        num_classes = net_cfg.NUM_CLASSES
        out_res = net_cfg.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.backbone(x)  # [bs, c, 8, 8]
        if self.neck is not None:
            conv_feat = self.neck(conv_feat)
        (
            vis_mask,
            full_mask,
            vis_vf,
            full_vf,
            vis_norm,
            full_norm,
            coor_x,
            coor_y,
            coor_z,
            region,
        ) = self.geo_head_net(conv_feat)

        # mask attentions
        if pnp_net_cfg.MASK_ATTENTION:
            vis_mask_atten = get_mask_prob(
                vis_mask, mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE
            )
            full_mask_atten = get_mask_prob(
                full_mask, mask_loss_type=net_cfg.LOSS_CFG.MASK_LOSS_TYPE
            )
        else:
            vis_mask_atten = vis_mask
            full_mask_atten = full_mask

        # 64region attentions
        if pnp_net_cfg.REGION_ATTENTION:
            # NOTE: for region, the 1st dim is bg
            region_atten = F.softmax(region[:, 1:, :, :], dim=1)
        else:
            region_atten = region

        out_dict = {
            "vis_mask_prob": vis_mask_atten,
            "full_mask_prob": full_mask_atten,
            "vis_vf": vis_vf,
            "full_vf": full_vf,
            "vis_norm": vis_norm,
            "full_norm": full_norm,
            "coor_x": coor_x,
            "coor_y": coor_y,
            "coor_z": coor_z,
            "region": region_atten,
        }
        if forward_mode == "geo":
            return out_dict

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat(
                [coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1
            )
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if pnp_net_cfg.WITH_2D_COORD:
            if pnp_net_cfg.COORD_2D_TYPE == "rel":
                assert roi_coord_2d_rel is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d_rel], dim=1)
            else:  # default abs
                assert roi_coord_2d is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat=coor_feat,
            region=region_atten,
            mask=torch.cat((full_mask_atten, vis_mask_atten), dim=1),
            norm=torch.cat((full_norm, vis_norm), dim=1),
            vf=torch.cat((full_vf, vis_vf), dim=1),
            extents=roi_extents,
        )

        # convert pred_rot to rot mat -------------------------
        rot_type = pnp_net_cfg.ROT_TYPE
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)

        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if pnp_net_cfg.TRANS_TYPE == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=pnp_net_cfg.Z_TYPE,
                # is_train=True
                is_train=self.training,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=self.training,  # TODO: sometimes we need it to be differentiable during test
            )
        elif pnp_net_cfg.TRANS_TYPE == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m,
                pred_t_,
                eps=1e-4,
                is_allo="allo" in rot_type,
                is_train=self.training,
            )
        else:
            raise ValueError(f"Unknown trans type: {pnp_net_cfg.TRANS_TYPE}")
        # endregion

        assert forward_mode == "pose", "Unknown forward_mode: {}".format(forward_mode)
        out_dict.update(
            {
                "rot": pred_ego_rot,
                "trans": pred_trans
            }
        )
        if gt_trans is not None and gt_ego_rot is not None:
            mean_re, mean_te = compute_mean_re_te(
                pred_trans, pred_rot_m, gt_trans, gt_ego_rot
            )
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(
                    pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()
                )
                * 100,  # cm
                "vis/error_ty": np.abs(
                    pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()
                )
                * 100,  # cm
                "vis/error_tz": np.abs(
                    pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()
                )
                * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
            }

            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)
        return out_dict


def build_model_optimizer(cfg, renderer=None, render_models=None, is_test=False):
    net_cfg = cfg.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    params_lr_list = []
    # backbone
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = BACKBONES[backbone_type](**init_backbone_args)
    if backbone_cfg.FREEZE:
        for param in backbone.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, backbone.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # neck --------------------------------
    neck, neck_params = get_neck(cfg)
    params_lr_list.extend(neck_params)

    # geo head -----------------------------------------------------
    geo_head, geo_head_params = get_geo_head(cfg)
    params_lr_list.extend(geo_head_params)

    # pnp net -----------------------------------------------
    pnp_net, pnp_net_params = get_pnp_net(cfg)
    params_lr_list.extend(pnp_net_params)

    # renderer -----------------------------------------------

    # build model
    model = GDRN_MaskNormVF(
        cfg, backbone, neck=neck, geo_head_net=geo_head, pnp_net=pnp_net, renderer=renderer, render_models=render_models
    )
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [
                        _param
                        for _name, _param in model.named_parameters()
                        if "log_var" in _name
                    ],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info(
                "Check if the backbone has been initialized with its own method!"
            )
            if backbone_pretrained == "timm":
                if init_backbone_args.pretrained and init_backbone_args.in_chans != 3:
                    load_timm_pretrained(
                        model.backbone,
                        in_chans=init_backbone_args.in_chans,
                        adapt_input_mode="custom",
                        strict=False,
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            # initialize backbone with official weights
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(
                model.backbone, backbone_pretrained, strict=False, logger=logger
            )
            logger.info(f"load backbone weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer


def _cosine_similarity_loss_vf(
    out_vf: torch.Tensor, gt_vf: torch.Tensor, mask
):
    b, c, _, w, h = out_vf.shape

    # without "detach()" may produce "RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED"
    # num_foreground_pix = torch.count_nonzero(gt_vf.detach()).item()
    num_foreground_pix = torch.count_nonzero(mask.detach()).item() * c
    cs_vf = F.cosine_similarity(out_vf, gt_vf, dim=2)  # b, c, w, h

    minus_cs_vf = torch.ones_like(cs_vf) - cs_vf

    masked_minus_cs_vf = mask.squeeze(2) * minus_cs_vf

    return masked_minus_cs_vf.sum() / num_foreground_pix
