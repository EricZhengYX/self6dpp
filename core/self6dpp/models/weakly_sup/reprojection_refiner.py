import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import roi_align

from core.self6dpp.losses.rot_loss import angular_distance_rot
from core.self6dpp.losses.pm_loss import PyPMLoss
from core.self6dpp.losses.mask_iou_loss import MaskIOULoss
from core.self6dpp.losses.bbox_iou_loss import IOULoss
from core.self6dpp.losses.ssim import MS_SSIM
from core.self6dpp.losses.perceptual_loss import PerceptualLoss
from lib.dr_utils.dib_renderer_x.renderer_dibr import Renderer_dibr as Render

from lib.pysixd.pose_error import add, adi

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")

HEIGHT = 480
WIDTH = 640
DEBUG = False
models_dir = "datasets/BOP_DATASETS/ycbv/models"

logger = logging.getLogger(__name__)


class RepjRefiner(nn.Module):
    def __init__(self, cfg, render: Render):
        super().__init__()
        assert cfg.REFINER.DO_REFINE, "Rep head not enabled"

        # configs
        self.cfg = cfg

        # RENDER
        self.render = render

        # Device
        self.device = "cuda"  # cfg.MODEL.DEVICE
        # Models & Models_info
        self.models = self.render.get_models()
        self.raw_models_info = self.render.get_models_info()
        self.models_info = {
            name: self._get_model_info_tensor(d)
            for name, d in self.raw_models_info.items()
        }

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
        self.mask_iou_loss = MaskIOULoss(
            reduction="mean"
        )
        self.perceptual_loss = PerceptualLoss()
        # self.ssim_loss_1 = MS_SSIM(channel=1)
        self.ssim_loss_3 = MS_SSIM(channel=3).cuda()

        # other things
        self.batch_size = int(cfg.SOLVER.IMS_PER_BATCH)
        self.multi_scale_miou = {
            1: 1,
            2: 2,
            4: 4,
        }

    def forward(self, inf_dict, ori_batch, seq_name):
        _set_roi_cls = set(ori_batch["roi_cls"].tolist())
        assert len(_set_roi_cls) == 1, ori_batch["roi_cls"].tolist()

        # self.dib_ren.set_camera_parameters_from_RT_K(Rs, ts, Ks, height=480, width=640, near=0.01, far=10, rot_type="mat")

        inf_r = inf_dict["rot"]
        inf_t = inf_dict["trans"].unsqueeze(2)
        assert inf_t.shape[0] == inf_r.shape[0]
        inf_r, inf_t = inf_r.to(self.device), inf_t.to(self.device)

        # TODO: Now actually sym_infos is not using
        sym_infos = ori_batch.get("sym_info", None)

        # The rendering can only be done on cuda
        batch_size = self.batch_size
        loss_dict = {}
        out_dict = {}
        record_dict = {}
        _cam_paras = {}
        rendering_bank = {}

        # camera intrinsic
        Ks = ori_batch["roi_cam"]

        # First, make rendering for scale 1
        roi_cs = ori_batch["roi_center"] / self.render.shrink
        roi_whs = ori_batch["roi_wh"] / self.render.shrink
        ori_roi = _bgr2rgb(ori_batch["roi_img"])
        ren_img_inf, ren_prob_mask_inf, _, _ = self.render(inf_r, inf_t, seq_name, Ks)
        ren_img_inf = ren_img_inf.permute(0, 3, 1, 2)
        ren_prob_mask_inf = ren_prob_mask_inf.permute(0, 3, 1, 2)
        # ren_prob_mask_inf_crop = self.crop_resize_tensor(
        #     ren_prob_mask_inf, roi_cs, roi_whs, out_size=ori_roi.shape[-2:]
        # )
        ren_img_inf_crop = self.crop_resize_tensor(
            ren_img_inf, roi_cs, roi_whs, out_size=ori_roi.shape[-2:]
        )

        # region self loss
        pseudo_vis_mask = (inf_dict["vis_mask"] > 0.5).to(torch.float32)
        pseudo_vis_mask = F.interpolate(
            pseudo_vis_mask,
            size=ori_roi.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss_dict["self_loss ms-ssim"] = (
            1 - self.ssim_loss_3(
                ren_img_inf_crop * pseudo_vis_mask,
                ori_roi * pseudo_vis_mask
            )
        ).mean() * self.weights.SELF_MSSSIM
        loss_dict["self_loss perceptual"] = self.perceptual_loss(
            ren_img_inf_crop * pseudo_vis_mask,
            ori_roi * pseudo_vis_mask
        ) * self.weights.SELF_PERCEP
        # endregion

        full_mask_vec = inf_dict["full_mask"].view(batch_size, -1).cpu().detach()
        vis_mask_vec = inf_dict["vis_mask"].view(batch_size, -1).cpu().detach()
        sim_scores = F.cosine_similarity(full_mask_vec, vis_mask_vec).numpy()

        best_inf_idx = np.argmax(sim_scores)

        record_dict["best idx"] = int(best_inf_idx)
        record_dict["sim_scores"] = sim_scores.tolist()
        rendering_bank[self.rendering_name_str()] = ren_img_inf
        rendering_bank[self.rendering_name_str(itype="pmsk")] = ren_prob_mask_inf

        # ==================== calculate & concat repj and inf paras ====================
        # region Rep&Inf-paras
        gt_pose = torch.cat(
            (
                torch.cat(
                    (ori_batch["ego_rot"], ori_batch["trans"].unsqueeze(2)), dim=-1
                ),
                torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=self.device).repeat(
                    batch_size, 1, 1
                ),
            ),
            dim=1,
        ).detach()
        rep_poses = torch.tensor((), device=self.device)
        rep_K = torch.tensor((), device=self.device)

        for i in range(batch_size):
            if i == best_inf_idx:
                continue

            # =========== Repj pose ===========
            this_cam_para = gt_pose[best_inf_idx] @ torch.inverse(gt_pose[i])
            tgt_inf_pose = torch.cat(
                (
                    torch.cat((inf_r[i], inf_t[i].view(3, 1)), dim=-1),
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device),
                ),
                dim=0,
            )
            src_rep_pose = this_cam_para @ tgt_inf_pose
            rep_poses = torch.cat((rep_poses, src_rep_pose.unsqueeze(0)))
            rep_K = torch.cat((rep_K, Ks[i].unsqueeze(0)))
        # endregion

        # =========== Disentangle R&T ===========
        # region RT
        rep_rs, rep_ts = rep_poses[:, :3, :3], rep_poses[:, :3, 3:]
        gt_r, gt_t = gt_pose[:, :3, :3], gt_pose[:, :3, 3:]
        # endregion

        # ==================== Which are pseudo labels? ====================
        # region pseudo-labels
        best_inf_r = inf_r[best_inf_idx].unsqueeze(0).detach()
        best_inf_t = inf_t[best_inf_idx].unsqueeze(0).detach()
        inf_rs = best_inf_r.repeat(batch_size - 1, 1, 1)
        inf_ts = best_inf_t.repeat(batch_size - 1, 1, 1)
        inf_K = Ks[best_inf_idx].unsqueeze(0).repeat(batch_size - 1, 1, 1)
        # endregion

        # ==================== Mask IOU loss + SSIM loss ====================
        # region miou+ssim
        valid_mious = []
        valid_miou_nums = {}
        for scale_name, scale in self.multi_scale_miou.items():
            if scale == 1:
                img_inf = (
                    rendering_bank[self.rendering_name_str()][best_inf_idx]
                    .unsqueeze(0)
                    .repeat(batch_size - 1, 1, 1, 1)
                    .detach()
                )  # b - 1, c, w, h
                mask_inf = (
                    rendering_bank[self.rendering_name_str(itype="pmsk")][best_inf_idx]
                    .squeeze(1)
                    .repeat(batch_size - 1, 1, 1)
                    .detach()
                )  # b - 1, w, h
            else:
                ren_img_inf, ren_prob_mask_inf, _, _ = self.render(
                    best_inf_r,
                    best_inf_t / scale,
                    seq_name,
                    Ks[best_inf_idx].unsqueeze(0),
                )
                img_inf = ren_img_inf.repeat(batch_size - 1, 1, 1, 1).permute(
                    0, 3, 1, 2
                )
                mask_inf = ren_prob_mask_inf.squeeze(3).repeat(batch_size - 1, 1, 1)
            ms_rep_ts = rep_ts / scale
            ren_img_rep, ren_prob_mask_rep, _, _ = self.render(
                rep_rs, ms_rep_ts, seq_name, rep_K
            )
            mask_rep = ren_prob_mask_rep.squeeze(3)
            img_rep = ren_img_rep.permute(0, 3, 1, 2)

            name_in_loss_dict = "mask iou {}".format(scale_name)
            loss_dict[name_in_loss_dict], this_valid_miou = self.mask_iou_loss(
                mask_inf, mask_rep
            ) * self.weights.MIOU
            valid_miou_nums.update(
                {
                    scale_name: int(sum(this_valid_miou)),
                }
            )
            valid_mious.append(this_valid_miou)

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
                    self.models[seq_name]["vertices"].repeat(rep_poses.shape[0], 1, 1),
                    inf_ts.squeeze(2),
                    rep_ts.squeeze(2),
                    sym_infos=sym_infos,
                ).values()
            )
        ) * self.weights.PM
        # endregion

        # ==================== 3D-2D GIOU loss ====================
        # region 3d2d-giou
        bboxes_rep_3d2d = self._get_boxes_from_rt(rep_rs, rep_ts, seq_name, rep_K)
        bboxes_inf_3d2d = self._get_boxes_from_rt(inf_rs, inf_ts, seq_name, inf_K)
        loss_dict["3d2d iou"] = (
            self.bbox_iou_loss(bboxes_inf_3d2d, bboxes_rep_3d2d) * 0.1
        ) * self.weights.IOU2D3D
        # endregion

        # ==================== Outputs & Record ====================
        # assert all(x.requires_grad for x in loss_dict.values()), 'Not all the losses are requiring grad!'
        # region SaveSomething
        record_dict["seq name"] = seq_name
        record_dict["valid_miou_nums"] = valid_miou_nums

        record_dict["rep r"] = rep_rs.tolist()
        record_dict["rep t"] = rep_ts.tolist()
        record_dict["inf r"] = inf_r.tolist()
        record_dict["inf t"] = inf_t.tolist()
        record_dict["gt r"] = gt_r.tolist()
        record_dict["gt t"] = gt_t.tolist()

        out_dict["rgt"] = self.criterion_r(gt_r, inf_r).item() / batch_size
        out_dict["tgt"] = self.criterion_t(gt_t, inf_t).item() / batch_size
        out_dict["rrep"] = self.criterion_r(rep_rs, inf_rs).item() / rep_rs.shape[0]
        out_dict["trep"] = self.criterion_t(rep_ts, inf_ts).item() / rep_ts.shape[0]

        sym = ori_batch["sym_info"][0] is not None
        out_dict["add_rep"] = self._make_add(
            inf_rs.cpu(),
            inf_ts.cpu(),
            rep_rs.cpu().detach(),
            rep_ts.cpu().detach(),
            seq_name,
            sym,
        ).mean()
        out_dict["add_gt"] = self._make_add(
            inf_r.cpu().detach(),
            inf_t.cpu().detach(),
            gt_r.cpu(),
            gt_t.cpu(),
            seq_name,
            sym,
        ).mean()
        # endregion

        return out_dict, loss_dict, record_dict

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad()
        self.ren.zero_grad()

    """
    def fetch_all_models(self, dir: str):
        pkl_name = "models_loaded_ply.pkl"
        pkl_dir = osp.join(dir, pkl_name)
        if osp.exists(pkl_dir):
            with open(pkl_dir, "br") as f:
                pkl = pickle.load(f)
            return pkl
        logger.info("Fetching all models from: {}".format(dir))
        _d = {}
        for name in LM_13_OBJECTS:
            _id = SEQ_DICT[name]
            _d[name] = load_plys(
                ply_paths=osp.join(dir, "obj_{}.ply".format(str(_id).zfill(6))),
                device=self.device,
            )[0]
        with open(pkl_dir, "bw") as f:
            pickle.dump(_d, f)
        return _d

    def fetch_model_infos(self, dir: str):
        json_name = "models_info.json"
        json_dir = osp.join(dir, json_name)
        assert osp.exists(json_dir), json_dir
        logger.info("Fetching all models_info from: {}".format(dir))
        with open(json_dir, "r") as f:
            info = json.load(f)
        _d = {}
        for name in LM_13_OBJECTS:
            _id = str(SEQ_DICT[name])
            _d[name] = {
                "diameter": info[_id]["diameter"] / 1000,
                "min_x": info[_id]["min_x"] / 1000,
                "min_y": info[_id]["min_y"] / 1000,
                "min_z": info[_id]["min_z"] / 1000,
                "size_x": info[_id]["size_x"] / 1000,
                "size_y": info[_id]["size_y"] / 1000,
                "size_z": info[_id]["size_z"] / 1000,
            }
        return _d
    """

    def _make_add(self, inf_r, inf_t, gt_r, gt_t, model_name, sym):
        model = self.models[model_name]["vertices"][0].cpu()
        diameter = self.raw_models_info[model_name]["diameter"]
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

    def _back_proj(self, points3d: torch.Tensor, K: torch.Tensor):
        """

        @param points3d: n*3*8
        @param K: n*3*3
        @return: n*3*8
        """
        assert points3d.ndim == 3 and points3d.shape[1] == 3
        z = points3d[:, 2, None]
        return torch.bmm(K, points3d) / z

    def _get_loss_weight(self, cfg):
        return cfg.REFINER.REFINER_LW

    def _get_boxes_from_rt(
        self, r: torch.Tensor, t: torch.Tensor, seq_name, K: torch.Tensor
    ):
        """

        @param r: n*3*3
        @param t: n*3*1
        @param seq_name: str
        @param K: n*3*3
        @return: n*4
        """
        b = r.shape[0]
        assert r.shape == (b, 3, 3) and t.shape == (b, 3, 1)
        assert r.device == t.device
        bbox_3d = r @ self.models_info[seq_name] + t  # n*3*8
        bbox_2d = self._back_proj(bbox_3d, K)  # n*3*8

        bbox_2d_x, bbox_2d_y = bbox_2d[:, 0, :], bbox_2d[:, 1, :]
        xmin = bbox_2d_x.min(dim=1, keepdim=True).values
        ymin = bbox_2d_y.min(dim=1, keepdim=True).values
        xmax = bbox_2d_x.max(dim=1, keepdim=True).values
        ymax = bbox_2d_y.max(dim=1, keepdim=True).values

        return torch.cat((xmin, ymin, xmax, ymax), dim=1)

    def rendering_name_str(self, ptype="inf", itype="rgb", mtype="batch", scale=1):
        """
        @param ptype: producing type 'inf' | 'repj'
        @param itype: image type 'rgb' | 'pmsk' | 'msk' | 'dpt'
        @param mtype: multiplication type 'batch' | 'multi'
        @param scale: int 1/2/4
        @return: a name string
        """
        if ptype in {"repj", "inf"}:
            return "{it}_{t}_{m}_scale{sc}".format(it=itype, t=ptype, m=mtype, sc=scale)
        else:
            raise ValueError(ptype)

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
