import logging
from functools import partial
from typing import List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import roi_align
from einops import rearrange
from detectron2.utils.events import get_event_storage

from core.self6dpp.losses.rot_loss import angular_distance_rot
from core.self6dpp.losses.pm_loss import PyPMLoss
from core.self6dpp.losses.mask_iou_loss import MaskIOULoss
from core.self6dpp.losses.bbox_iou_loss import IOULoss
from core.self6dpp.losses.ssim import MS_SSIM
from core.self6dpp.models.model_utils import compute_mean_re_te
from lib.pysixd.pose_error import add, adi
from lib.pysixd.misc import backproject_th

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
logger = logging.getLogger(__name__)


class RepjRefiner(nn.Module):
    def __init__(
        self,
        cfg,
        render: Callable,
        models: List[dict],
        models_info: List[dict],
        train_obj_names: List[str],
    ):
        super().__init__()
        assert cfg.REPJ_REFINE.ENABLE, "RepjRefiner is not enabled"

        # configs
        self.cfg = cfg

        # Rendering
        self.render_shrink = cfg.REPJ_REFINE.RENDERER.SHRINK
        self.ren_W = int(cfg.RENDERER.DIBR.WIDTH / self.render_shrink)
        self.ren_H = int(cfg.RENDERER.DIBR.HEIGHT / self.render_shrink)
        self.render = partial(
            render, width=self.ren_W, height=self.ren_H, mode=["color", "prob"]
        )
        # Device
        self.device = "cuda"  # cfg.MODEL.DEVICE
        # Models & Models_info
        self.obj_names = train_obj_names
        self.models = models
        self.raw_models_info = models_info
        self.models_info = [
            self._get_model_info_tensor(d) for d in self.raw_models_info
        ]
        assert len(self.obj_names) == len(self.models) == len(self.models_info)

        # Loss weights
        self.weights = self._get_loss_weight(cfg=cfg)
        # Loss module
        self.bbox_iou_loss = IOULoss(loss_type="giou")
        self.criterion_r = angular_distance_rot
        self.criterion_t = nn.SmoothL1Loss()
        self.pm_loss = PyPMLoss(
            loss_type="smooth_l1",
            disentangle_t=True,
            t_loss_use_points=True,
            r_only=False,
        )
        self.mask_iou_loss = MaskIOULoss(reduction="mean")
        # self.ssim_loss_1 = MS_SSIM(channel=1).cuda()
        self.ssim_loss_3 = MS_SSIM(channel=3).cuda()

        # other things
        self.batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
        self.multi_scale_miou = {
            1: 1,
            2: 2,
            4: 4,
        }

    def forward(
        self,
        gt_pose,
        inf_rot,
        inf_trans,
        inf_full_masks,
        inf_vis_masks,
        roi_cls,
        roi_centers,
        roi_whs,
        cam_K,
        sym_infos=None,
    ):
        """

        @param gt_pose: b*3*4
        @param inf_rot: b*3*3
        @param inf_trans: b*3*1
        @param inf_full_masks: b*64*64
        @param inf_vis_masks: b*64*64
        @param roi_cls: b
        @param roi_centers: b*2
        @param roi_whs: b*2
        @param cam_K: b*3*3
        @param sym_infos: b*3*3/None
        @return: ws_out_dict, ws_loss_dict, ws_record_dict
        """
        assert len(roi_cls.unique()) == 1, "When using RepjRefiner, cls in one batch should be consist"
        assert (
            inf_rot.shape[0]
            == inf_trans.shape[0]
            == roi_cls.shape[0]
            == roi_centers.shape[0]
            == roi_whs.shape[0]
        )

        # The rendering can only be done on cuda
        batch_size = self.batch_size
        loss_dict = {}
        vis_dict = {}
        record_dict = {}
        _cam_paras = {}

        # basic renderer settings
        roi_cls_id = roi_cls[0]
        roi_cs = roi_centers / self.render_shrink
        roi_whs = roi_whs / self.render_shrink
        cur_models = [self.models[int(_l)] for _l in roi_cls]

        # best view point
        full_mask_vec = inf_full_masks.view(batch_size, -1).cpu().detach()
        full_mask_vec_hard = full_mask_vec > 0.5
        vis_mask_vec = inf_vis_masks.view(batch_size, -1).cpu().detach()
        sim_scores = []
        for i in range(batch_size):
            _fvec = full_mask_vec[i]
            _fvech = full_mask_vec_hard[i]
            _vvec = vis_mask_vec[i]
            sim_scores.append(
                F.cosine_similarity(_fvec[_fvech], _vvec[_fvech], dim=0).numpy()
            )
        best_inf_idx = np.argmax(sim_scores)

        record_dict["best idx"] = int(best_inf_idx)
        record_dict["sim_scores"] = sim_scores.tolist()

        # ==================== calculate & concat repj and inf paras ====================
        # region Rep&Inf-paras
        gt_pose = torch.cat(
            (
                gt_pose,
                torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=self.device).repeat(
                    batch_size, 1, 1
                ),
            ),
            dim=1,
        ).detach()  # b*4*4
        rep_poses = torch.tensor((), device=self.device)
        rep_K = torch.tensor((), device=self.device)
        rep_models = []

        for i in range(batch_size):
            if i == best_inf_idx:
                continue
            # =========== Repj pose ===========
            this_cam_para = gt_pose[best_inf_idx] @ torch.inverse(gt_pose[i])
            tgt_inf_pose = torch.cat(
                (
                    torch.cat((inf_rot[i], inf_trans[i]), dim=-1),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device),
                ),
                dim=0,
            )
            src_rep_pose = this_cam_para @ tgt_inf_pose
            rep_poses = torch.cat((rep_poses, src_rep_pose.unsqueeze(0)))
            rep_K = torch.cat((rep_K, cam_K[i].unsqueeze(0)))
            rep_models.append(cur_models[i])
        # endregion

        # =========== Disentangle R&T ===========
        # region RT
        rep_rs, rep_ts = rep_poses[:, :3, :3], rep_poses[:, :3, 3:]
        gt_r, gt_t = gt_pose[:, :3, :3], gt_pose[:, :3, 3:]
        # endregion

        # ==================== Which are pseudo labels? ====================
        # region pseudo-labels
        best_inf_r = inf_rot[best_inf_idx].unsqueeze(0).detach()
        best_inf_t = inf_trans[best_inf_idx].unsqueeze(0).detach()
        inf_rs = best_inf_r.repeat(batch_size - 1, 1, 1)
        inf_ts = best_inf_t.repeat(batch_size - 1, 1, 1)
        inf_K = cam_K[best_inf_idx].unsqueeze(0).repeat(batch_size - 1, 1, 1)
        # endregion

        # ==================== Mask IOU loss + SSIM loss ====================
        # region miou+ssim
        for scale_name, scale in self.multi_scale_miou.items():
            # images by rendering inf-poses
            inf_ren_dict = self.render(
                best_inf_r,
                best_inf_t / scale,
                cur_models[best_inf_idx: best_inf_idx + 1],
                Ks=cam_K[best_inf_idx].unsqueeze(0),
            )
            img_inf = rearrange(
                inf_ren_dict["color"], "b h w c -> b c h w"
            ).repeat(batch_size - 1, 1, 1, 1)
            mask_inf = inf_ren_dict["prob"].repeat(batch_size - 1, 1, 1)  # (b-1)*h*w

            # images by rendering repj-poses
            ms_rep_ts = rep_ts / scale
            rep_ren_dict = self.render(
                rep_rs,
                ms_rep_ts / scale,
                rep_models,
                Ks=rep_K,
            )
            img_rep = rearrange(
                rep_ren_dict["color"], "b h w c -> b c h w"
            )
            mask_rep = rep_ren_dict["prob"]  # (b-1)*h*w

            name_in_loss_dict = "mask iou {}".format(scale_name)
            loss_dict[name_in_loss_dict], this_valid_miou = (
                self.mask_iou_loss(mask_inf, mask_rep) * self.weights.MIOU
            )

            # ===== MS SSIM =====
            name_in_loss_dict = "ms ssim {}".format(scale_name)
            if img_rep.shape[0]:
                loss_dict[name_in_loss_dict] = (
                    1 - self.ssim_loss_3(img_inf, img_rep)
                ).mean() * self.weights.MSSSIM
            else:
                loss_dict[name_in_loss_dict] = 0
        # endregion

        # ==================== PM loss ====================
        # region PM
        loss_dict["pm"] = (
            sum(
                self.pm_loss(
                    inf_rs,
                    rep_rs,
                    cur_models[best_inf_idx]["vertices"].repeat(
                        rep_poses.shape[0], 1, 1
                    ),
                    inf_ts.squeeze(2),
                    rep_ts.squeeze(2),
                    sym_infos=sym_infos,
                ).values()
            )
        ) * self.weights.PM
        # endregion

        # ==================== 3D-2D GIOU loss ====================
        # region 3d2d-giou
        bboxes_rep_3d2d = self._get_boxes_from_rt(rep_rs, rep_ts, roi_cls_id, rep_K)
        bboxes_inf_3d2d = self._get_boxes_from_rt(inf_rs, inf_ts, roi_cls_id, inf_K)
        loss_dict["3d2d iou"] = (
            self.bbox_iou_loss(bboxes_inf_3d2d, bboxes_rep_3d2d) * 0.1
        ) * self.weights.IOU2D3D
        # endregion

        # ==================== Outputs & Record ====================
        # region SaveSomething
        record_dict["seq name"] = self.obj_names[roi_cls_id]

        record_dict["rep r"] = rep_rs.tolist()
        record_dict["rep t"] = rep_ts.tolist()
        record_dict["inf r"] = inf_rot.tolist()
        record_dict["inf t"] = inf_trans.tolist()
        record_dict["gt r"] = gt_r.tolist()
        record_dict["gt t"] = gt_t.tolist()

        '''
        # gt errors should be already computed in GDRN
        r_error_gt, t_error_gt = compute_mean_re_te(inf_trans, inf_rot, gt_t, gt_r)
        vis_dict["r_error_gt"] = float(r_error_gt)
        vis_dict["t_error_gt"] = float(t_error_gt)
        '''
        r_error_rp, t_error_rp = compute_mean_re_te(inf_ts, inf_rs, rep_ts, rep_rs)
        vis_dict["r_error_rp"] = float(r_error_rp)
        vis_dict["t_error_rp"] = float(t_error_rp)

        sym = sym_infos[0] is not None
        vis_dict["add_rp"] = self._make_add(
            inf_rs.cpu(),
            inf_ts.cpu(),
            rep_rs.cpu().detach(),
            rep_ts.cpu().detach(),
            model_vertices=cur_models[0]["vertices"],
            diameter=self.raw_models_info[roi_cls_id]["diameter"],
            sym=sym,
        ).mean()
        vis_dict["add_gt"] = self._make_add(
            inf_rot.cpu().detach(),
            inf_trans.cpu().detach(),
            gt_r.cpu(),
            gt_t.cpu(),
            model_vertices=cur_models[0]["vertices"],
            diameter=self.raw_models_info[roi_cls_id]["diameter"],
            sym=sym,
        ).mean()

        storage = get_event_storage()
        for k, v in vis_dict.items():
            storage.put_scalar("vis/" + k, v)
        # endregion

        return vis_dict, loss_dict, record_dict

    def _make_add(self, inf_r, inf_t, gt_r, gt_t, model_vertices, diameter, sym):
        model = model_vertices.cpu()
        add_func = adi if sym else add

        bs = inf_r.shape[0]

        res = []
        for i in range(bs):
            pred_r = inf_r[i]
            pred_t = inf_t[i]
            gt_r_ = gt_r[i]
            gt_t_ = gt_t[i]
            res.append(
                add_func(
                    pred_r.numpy(),
                    pred_t.numpy(),
                    gt_r_.numpy(),
                    gt_t_.numpy(),
                    pts=model.numpy(),
                )
                / diameter
                * 100
            )

        return np.array(res)

    def _get_model_info_tensor(self, model_info_dict: dict) -> torch.Tensor:
        assert "min_x" in model_info_dict
        min_x = model_info_dict["min_x"]
        assert "min_y" in model_info_dict
        min_y = model_info_dict["min_y"]
        assert "min_z" in model_info_dict
        min_z = model_info_dict["min_z"]
        assert "size_x" in model_info_dict
        size_x = model_info_dict["size_x"]
        assert "size_y" in model_info_dict
        size_y = model_info_dict["size_y"]
        assert "size_z" in model_info_dict
        size_z = model_info_dict["size_z"]
        model_3dbbox = torch.tensor(
            [
                [min_x, min_y, min_z],
                [min_x + size_x, min_y, min_z],
                [min_x + size_x, min_y + size_y, min_z],
                [min_x, min_y + size_y, min_z],
                [min_x, min_y, min_z + size_z],
                [min_x + size_x, min_y, min_z + size_z],
                [min_x + size_x, min_y + size_y, min_z + size_z],
                [min_x, min_y + size_y, min_z + size_z],
            ]
        ).T.unsqueeze(0)
        assert model_3dbbox.shape == (1, 3, 8)
        return model_3dbbox.to(self.device)

    def _get_loss_weight(self, cfg):
        return cfg.REPJ_REFINE.REPJ_REFINER_LW

    def _proj_3Dbbox(self, points3d: torch.Tensor, K: torch.Tensor):
        """

        @param points3d: n*3*8
        @param K: n*3*3
        @return: n*3*8
        """
        assert points3d.ndim == 3 and points3d.shape[1] == 3
        z = points3d[:, 2, None]
        return torch.bmm(K, points3d) / z

    def _get_boxes_from_rt(
        self, r: torch.Tensor, t: torch.Tensor, cls_id, K: torch.Tensor
    ):
        """

        @param r: n*3*3
        @param t: n*3*1
        @param cls_id: int
        @param K: n*3*3
        @return: n*4
        """
        b = r.shape[0]
        assert r.shape == (b, 3, 3) and t.shape == (b, 3, 1)
        assert r.device == t.device
        bbox_3d = r @ self.models_info[cls_id] + t  # n*3*8
        bbox_2d = self._proj_3Dbbox(bbox_3d, K)  # n*3*8

        bbox_2d_x, bbox_2d_y = bbox_2d[:, 0, :], bbox_2d[:, 1, :]
        xmin = bbox_2d_x.min(dim=1, keepdim=True).values
        ymin = bbox_2d_y.min(dim=1, keepdim=True).values
        xmax = bbox_2d_x.max(dim=1, keepdim=True).values
        ymax = bbox_2d_y.max(dim=1, keepdim=True).values

        return torch.cat((xmin, ymin, xmax, ymax), dim=1)

    def crop_resize_tensor(
        self, m: torch.Tensor, center: torch.Tensor, wh: torch.Tensor, out_size=(64, 64)
    ):
        batch_size = m.shape[0]
        idx_tensor = torch.tensor(range(batch_size)).to(m)
        a_05 = wh.max(1).values * self.cfg.INPUT.DZI_PAD_SCALE / 2
        l, t, r, b = (
            center[:, 0] - a_05,
            center[:, 1] - a_05,
            center[:, 0] + a_05,
            center[:, 1] + a_05,
        )

        boxes_with_idx = torch.stack((idx_tensor, l, t, r, b)).T
        return roi_align(input=m, boxes=boxes_with_idx.to(m), output_size=out_size)

    def batch2multi(self, m: torch.Tensor):
        assert m.ndim == 3 and (
            m.shape[1:] == (3, 3) or m.shape[1:] == (3, 1) or m.shape[1:] == (4, 4)
        ), m.shape
        b, x, y = m.shape
        return m[:, None, :, :].expand(b, b - 1, x, y).reshape(b * (b - 1), x, -1)

    def get_render(self):
        return self.render

    def get_models(self):
        return self.models


def _bgr2rgb(img: torch.Tensor):
    assert img.ndim == 4 and img.shape[1] == 3, img.shape
    b = img[:, :1, :, :]
    g = img[:, 1:2, :, :]
    r = img[:, 2:, :, :]
    return torch.cat((r, g, b), dim=1)


def build_repj_refiner(
    cfg, renderer, render_models, data_ref, train_objs
) -> RepjRefiner:
    # get model_info
    all_model_infos = data_ref.get_models_info()
    models_info = [
        all_model_infos[str(data_ref.obj2id[obj_name])] for obj_name in train_objs
    ]
    return RepjRefiner(
        cfg=cfg,
        render=renderer,
        models=render_models,
        models_info=models_info,
        train_obj_names=train_objs,
    )
