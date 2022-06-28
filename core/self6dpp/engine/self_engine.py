import logging
import os
import os.path as osp
import torch
from torch import nn
import mmcv
from mmcv.runner.checkpoint import load_checkpoint
import time
import cv2
import numpy as np
from collections import OrderedDict

from detectron2.utils.events import EventStorage
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)

from detectron2.data.common import AspectRatioGroupedDataset
from detectron2.data import MetadataCatalog, DatasetCatalog

from lib.pysixd.pose_error import re, te, add
from lib.utils.utils import dprint
from lib.vis_utils.image import grid_show, vis_bbox_opencv
from lib.torch_utils.torch_utils import ModelEMA
from lib.torch_utils.misc import nan_to_num
from core.utils import solver_utils
import core.utils.my_comm as comm
from core.utils.my_checkpoint import MyCheckpointer
from core.utils.my_writer import (
    MyCommonMetricPrinter,
    MyJSONWriter,
    MyTensorboardXWriter,
)
from core.utils.utils import get_emb_show
from core.utils.data_utils import denormalize_image
from core.self6dpp.datasets.data_loader_self import build_gdrn_self_train_loader
from core.self6dpp.datasets.data_loader import (
    build_gdrn_train_loader,
    build_gdrn_test_loader,
)
from core.self6dpp.losses.ssim import SSIM, MS_SSIM
from core.self6dpp.losses.vf_norm_loss import VFLoss, NORMLoss
from core.self6dpp.losses.perceptual_loss import PerceptualLoss

from core.self6dpp.models.weakly_sup.reprojection_refiner import (
    build_repj_refiner,
    RepjRefiner,
)

from .gdrn_engine_utils import batch_data, get_out_coor, get_out_mask
from .self_engine_utils import batch_data_self, compute_self_loss
from .gdrn_evaluator import (
    gdrn_inference_on_dataset,
    GDRN_Evaluator,
    gdrn_save_result_of_dataset,
)
from .gdrn_custom_evaluator import GDRN_EvaluatorCustom
import ref

logger = logging.getLogger(__name__)


def get_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator(s) for a given dataset.

    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = osp.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)

    _distributed = comm.get_world_size() > 1
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    train_obj_names = dataset_meta.objs
    if evaluator_type == "bop":
        gdrn_eval_cls = (
            GDRN_Evaluator if cfg.VAL.get("USE_BOP", False) else GDRN_EvaluatorCustom
        )
        return gdrn_eval_cls(
            cfg,
            dataset_name,
            distributed=_distributed,
            output_dir=output_folder,
            train_objs=train_obj_names,
        )

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_save_results(cfg, model, epoch=None, iteration=None):
    model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]

    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    train_obj_names = dataset_meta.objs

    for dataset_name in cfg.DATASETS.TEST:
        if epoch is not None and iteration is not None:
            save_out_dir = osp.join(
                cfg.OUTPUT_DIR,
                f"inference_epoch_{epoch}_iter_{iteration}",
                dataset_name,
            )
        else:
            save_out_dir = osp.join(
                cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name
            )

        data_loader = build_gdrn_test_loader(
            cfg, dataset_name, train_objs=train_obj_names
        )
        gdrn_save_result_of_dataset(
            cfg,
            model,
            data_loader,
            output_dir=save_out_dir,
            dataset_name=dataset_name,
            train_objs=train_obj_names,
            amp_test=cfg.TEST.AMP_TEST,
        )


def do_test(cfg, model, epoch=None, iteration=None):
    results = OrderedDict()
    model_name = osp.basename(cfg.MODEL.WEIGHTS).split(".")[0]
    for dataset_name in cfg.DATASETS.TEST:
        if epoch is not None and iteration is not None:
            eval_out_dir = osp.join(
                cfg.OUTPUT_DIR,
                f"inference_epoch_{epoch}_iter_{iteration}",
                dataset_name,
            )
        else:
            eval_out_dir = osp.join(
                cfg.OUTPUT_DIR, f"inference_{model_name}", dataset_name
            )
        evaluator = get_evaluator(cfg, dataset_name, eval_out_dir)
        data_loader = build_gdrn_test_loader(
            cfg, dataset_name, train_objs=evaluator.train_objs
        )
        results_i = gdrn_inference_on_dataset(
            cfg, model, data_loader, evaluator, amp_test=cfg.TEST.AMP_TEST
        )
        results[dataset_name] = results_i
        # if comm.is_main_process():
        #     logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        #     print_csv_format(results_i)
    # if len(results) == 1:
    #     results = list(results.values())[0]
    return results


def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
    return tbx_event_writer


def do_train(
    cfg,
    args,
    model,
    optimizer,
    model_teacher=None,
    repj_refine: RepjRefiner = None,
    refiner=None,
    ref_cfg=None,
    renderer_func=None,
    ren_models=None,
    resume=False,
):
    net_cfg = cfg.MODEL.POSE_NET
    self_loss_cfg = net_cfg.SELF_LOSS_CFG
    model.train()
    model_teacher.eval()
    if refiner is not None:
        refiner.eval()
    if repj_refine is not None:
        logger.warning("Also Weakly-supervised!")
        do_weakly = True
    else:
        do_weakly = False

    # some basic settings =========================
    dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_ref = ref.__dict__[dataset_meta.ref_key]
    obj_names = dataset_meta.objs

    # load data ===================================
    train_dset_names = cfg.DATASETS.TRAIN
    data_loader = build_gdrn_self_train_loader(
        cfg, train_dset_names, train_objs=obj_names
    )
    data_loader_iter = iter(data_loader)

    # load 2nd train dataloader if needed (assume this is synthetic data)
    train_2_dset_names = cfg.DATASETS.get("TRAIN2", ())
    train_2_ratio = cfg.DATASETS.get("TRAIN2_RATIO", 0.0)
    if train_2_ratio > 0.0 and len(train_2_dset_names) > 0:
        data_loader_2 = build_gdrn_train_loader(cfg, train_2_dset_names)
        data_loader_2_iter = iter(data_loader_2)
    else:
        data_loader_2 = None
        data_loader_2_iter = None

    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    if isinstance(data_loader, AspectRatioGroupedDataset):
        dataset_len = len(data_loader.dataset.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    else:
        dataset_len = len(data_loader.dataset)
        iters_per_epoch = dataset_len // images_per_batch
    # NOTE: here iters are based on real data!!!
    max_iter = cfg.SOLVER.TOTAL_EPOCHS * iters_per_epoch
    clip_grad = cfg.SOLVER.get("CLIP_GRAD", 100)

    dprint("images_per_batch: ", images_per_batch)
    dprint("dataset length: ", dataset_len)
    dprint("iters per epoch: ", iters_per_epoch)
    dprint("total iters: ", max_iter)

    bs_ref = cfg.SOLVER.get(
        "REFERENCE_BS", 64
    )  # nominal batch size =========================
    accumulate_iter = max(
        round(bs_ref / cfg.SOLVER.IMS_PER_BATCH), 1
    )  # accumulate loss before optimizing
    # NOTE: update lr every accumulate_iter
    scheduler = solver_utils.build_lr_scheduler(
        cfg, optimizer, total_iters=max_iter // accumulate_iter
    )

    # resume or load model ===================================
    # teacher/student initialized by the same weights (only load teacher for train)
    assert osp.exists(cfg.MODEL.WEIGHTS), cfg.MODEL.WEIGHTS
    checkpointer = MyCheckpointer(
        model,
        cfg.OUTPUT_DIR,
        model_teacher=model_teacher,  # NOTE: teacher
        optimizer=optimizer,
        scheduler=scheduler,
        save_to_disk=comm.is_main_process(),
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    start_epoch = start_iter // iters_per_epoch + 1  # first epoch is 1

    # initialize teacher model
    logger.info("load teacher weights from {}".format(cfg.MODEL.WEIGHTS))
    load_checkpoint(model_teacher, cfg.MODEL.WEIGHTS, logger=logger)

    # Exponential moving average for teacher (NOTE: initialize ema after loading weights) ========================
    if comm.is_main_process():
        ema = ModelEMA(model_teacher, **cfg.MODEL.EMA.INIT_CFG)
        ema.updates = start_epoch // cfg.MODEL.EMA.UPDATE_FREQ
        # save the ema model
        checkpointer.model = ema.ema.module if hasattr(ema.ema, "module") else ema.ema
    else:
        ema = None

    if comm._USE_HVD:  # hvd may be not available, so do not use the one in args
        import horovod.torch as hvd

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_parameters(model_teacher.state_dict(), root_rank=0)
        if refiner is not None:
            hvd.broadcast_parameters(refiner.state_dict(), root_rank=0)

        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Horovod: (optional) compression algorithm.
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        )
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            compression=compression,
        )  # device_dense='/cpu:0'

    if cfg.SOLVER.CHECKPOINT_BY_EPOCH:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD * iters_per_epoch
    else:
        ckpt_period = cfg.SOLVER.CHECKPOINT_PERIOD
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=cfg.SOLVER.MAX_TO_KEEP
    )
    eval_period = cfg.TEST.EVAL_PERIOD * iters_per_epoch

    # ------------------------------------------------------------------
    # init some loss funcs
    # ------------------------------------------------------------------
    ssim_func = SSIM(data_range=1.0).cuda()
    ms_ssim_func = MS_SSIM(data_range=1.0, normalize=True).cuda()
    vf_loss_func = VFLoss().cuda()
    norm_loss_func = NORMLoss().cuda()
    if self_loss_cfg.PERCEPT_LW > 0:
        percep_loss_func = PerceptualLoss(model="net", net="alex", use_gpu=True)
    else:
        percep_loss_func = None

    # ------------------------------------------------------------------
    # build writers
    # ------------------------------------------------------------------
    tbx_event_writer = get_tbx_event_writer(
        cfg.OUTPUT_DIR, backup=not cfg.get("RESUME", False)
    )
    tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
    writers = (
        {
            "terminal": MyCommonMetricPrinter(max_iter, eval_period, ckpt_period),
            "json": MyJSONWriter(osp.join(cfg.OUTPUT_DIR, "metrics.json")),
            "tbx": tbx_event_writer,
        }
        if comm.is_main_process()
        else dict()
    )

    if cfg.TRAIN.DEBUG_SINGLE_IM:
        # load gt pose
        gt_dict = DatasetCatalog.get(train_dset_names[0])[0]
        scene_im_id = gt_dict["scene_im_id"]
        im_id = scene_im_id.split("/")[1]
        assert im_id in train_dset_names[0], "{} {}".format(im_id, train_dset_names[0])
        gt_pose = gt_dict["annotations"][0]["pose"]

        debug_results = {}

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    logger.info("Starting training from iteration {}".format(start_iter))
    iter_time = None
    with EventStorage(start_iter) as storage:
        optimizer.zero_grad()
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration
            epoch = iteration // iters_per_epoch + 1  # epoch start from 1
            storage.put_scalar("epoch", epoch, smoothing_hint=False)

            is_log_iter = False
            if iteration - start_iter > 5 and (
                (iteration + 1) % cfg.TRAIN.PRINT_FREQ == 0
                or iteration == max_iter - 1
                or iteration < 100
            ):
                is_log_iter = True

            if cfg.TRAIN.DEBUG_SINGLE_IM:
                is_log_iter = True

            do_syn_sup = False
            do_self = False
            this_iter_forward_mode = ""  # 'pose' | 'geo'
            if np.random.rand() < train_2_ratio:  # synthetic supervised
                data_loader_2.dataset.step()
                data = next(data_loader_2_iter)
                do_syn_sup = True
                this_iter_forward_mode = data_loader_2.dataset.get_current_output_mode()
            else:
                data_loader.dataset.step()
                data = next(data_loader_iter)
                do_self = True
                this_iter_forward_mode = data_loader.dataset.get_current_output_mode()

            storage.put_scalar(
                "forward_mode",
                0 if this_iter_forward_mode == "pose" else 1,
                smoothing_hint=False,
            )

            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()

            # region forward
            # ------------------------------------------------------------------
            # forward
            # ------------------------------------------------------------------
            ws_loss_dict = dict()
            if do_syn_sup:  # (synthetic supervised batch)
                # NOTE: use offline xyz labels (DIBR rendered xyz is not very accurate)
                assert (
                    net_cfg.XYZ_ONLINE is False
                ), "Use offline xyz labels for self-supervised training!"
                batch = batch_data(cfg, data, renderer=None)
                out_dict, syn_loss_dict = model(
                    batch["roi_img"],
                    gt_xyz=batch.get("roi_xyz", None),
                    gt_xyz_bin=batch.get("roi_xyz_bin", None),
                    gt_mask_trunc=batch["roi_mask_trunc"],
                    gt_mask_visib=batch["roi_mask_visib"],
                    gt_mask_obj=batch["roi_mask_obj"],
                    gt_mask_full=batch.get("roi_mask_full", None),
                    gt_region=batch.get("roi_region", None),
                    gt_ego_rot=batch.get("ego_rot", None),
                    gt_trans=batch.get("trans", None),
                    gt_trans_ratio=batch["roi_trans_ratio"],
                    gt_points=batch.get("roi_points", None),
                    sym_infos=batch.get("sym_info", None),
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                    do_loss=True,
                )
                syn_losses = sum(syn_loss_dict.values())
                assert torch.isfinite(syn_losses).all(), syn_loss_dict

                loss_dict_reduced = {
                    k: v.item() for k, v in comm.reduce_dict(syn_loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                losses = syn_losses
            elif do_self:
                batch = batch_data_self(
                    cfg,
                    data,
                    loss_mode=this_iter_forward_mode,
                    model_teacher=model_teacher,
                )
                # only outputs, no losses
                out_dict = model(
                    batch["roi_img"],
                    gt_ego_rot=batch.get("gt_rot"),
                    gt_trans=batch.get("gt_trans"),
                    roi_coord_2d=batch.get("roi_coord_2d"),
                    roi_coord_2d_rel=batch.get("roi_coord_2d_rel"),
                    roi_cams=batch.get("roi_cam"),
                    roi_centers=batch.get("roi_center"),
                    roi_whs=batch.get("roi_wh"),
                    roi_extents=batch.get("roi_extent"),
                    resize_ratios=batch.get("resize_ratio"),
                    forward_mode=this_iter_forward_mode,
                )
                # compute self-supervised losses

                self_loss_dict = compute_self_loss(
                    cfg,
                    batch,
                    pred_rot=out_dict.get("rot", None),
                    pred_trans=out_dict.get("trans", None),
                    pred_vis_mask_prob=out_dict["vis_mask_prob"],
                    pred_full_mask_prob=out_dict["full_mask_prob"],
                    pred_coor_x=out_dict["coor_x"],
                    pred_coor_y=out_dict["coor_y"],
                    pred_coor_z=out_dict["coor_z"],
                    pred_region=out_dict["region"],
                    pred_vis_vf=out_dict["vis_vf"],
                    pred_full_vf=out_dict["full_vf"],
                    pred_vis_norm=out_dict["vis_norm"],
                    pred_full_norm=out_dict["full_norm"],
                    ren_func=renderer_func,
                    ren_models=ren_models,
                    loss_mode=this_iter_forward_mode,
                    vf_loss_func=vf_loss_func,
                    norm_loss_func=norm_loss_func,
                    ssim_func=ssim_func,
                    ms_ssim_func=ms_ssim_func,
                    perceptual_func=percep_loss_func,
                    tb_writer=tbx_writer if is_log_iter else None,
                    iteration=iteration if is_log_iter else None,
                )
                """
                finite_loss_dict = {
                    k: v for k, v in self_loss_dict.items() if torch.isfinite(v)
                }
                assert len(finite_loss_dict) > 0, finite_loss_dict
                if len(finite_loss_dict) < len(self_loss_dict):
                    failed_names = [
                        k for k, v in self_loss_dict.items() if not torch.isfinite(v)
                    ]
                    warn_msg = "following item(s) encountered NaN in iter{it}:\n{items}".format(
                        it=iteration, items="\n".join(failed_names)
                    )
                    logger.warning(warn_msg)
                """
                self_losses = sum(self_loss_dict.values())
                assert torch.isfinite(self_losses).all(), self_loss_dict

                loss_dict_reduced = {
                    k: v.item() for k, v in comm.reduce_dict(self_loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                # if needed, make weakly-supervised as well
                if do_weakly and this_iter_forward_mode == "pose":
                    ws_vis_dict, ws_loss_dict, ws_record_dict = repj_refine.forward(
                        gt_pose=batch["gt_pose_bx3x4"],
                        inf_rot=out_dict["rot"],
                        inf_trans=out_dict["trans"].unsqueeze(-1),
                        inf_full_masks=out_dict["full_mask_prob"],
                        inf_vis_masks=out_dict["vis_mask_prob"],
                        roi_cls=batch["roi_cls"],
                        roi_centers=batch["roi_center"],
                        roi_whs=batch["roi_wh"],
                        cam_K=batch["roi_cam"],
                        sym_infos=batch["sym_info"],
                    )
                    # TODO: FIX THIS UGLY THING
                    for k, v in ws_loss_dict.items():
                        if torch.isfinite(v):
                            ws_loss_dict[k] = 0
                    weakly_losses = sum(ws_loss_dict.values())
                    assert torch.isfinite(weakly_losses).all(), ws_loss_dict
                else:
                    weakly_losses = 0
                losses = self_losses + weakly_losses
                """
                if comm.is_main_process():
                    storage.put_scalars(
                        total_loss_self=losses_reduced, **loss_dict_reduced
                    )
                    if cfg.TRAIN.DEBUG_SINGLE_IM:
                        r_est = out_dict["rot"][0].detach().cpu().numpy()
                        t_est = out_dict["trans"][0].detach().cpu().numpy()
                        r_error = re(r_est, gt_pose[:3, :3])
                        t_error = te(t_est, gt_pose[:3, 3]) * 100
                        add_error = add(
                            r_est,
                            t_est,
                            gt_pose[:3, :3],
                            gt_pose[:3, 3],
                            batch["roi_points"][0].detach().cpu().numpy(),
                        )
                        debug_results["iter_{:04d}".format(iteration)] = {
                            "pose_gt": gt_pose,
                            "pose_est": np.hstack([r_est, t_est.reshape(3, 1)]),
                            "r_error": r_error,
                            "t_error": t_error,
                            "add_error": add_error,
                            "loss": losses_reduced,
                        }
                        storage.put_scalars(
                            r_error_self=r_error,
                            t_error_self=t_error,
                            add_error_self=add_error,
                        )
                """
            else:
                raise RuntimeError("Not in do_self or do_syn_sup")
            # endregion

            # region backward & optimize
            # ------------------------------------------------------------------
            # backward & optimize
            # ------------------------------------------------------------------
            losses.backward()
            # optimize
            if iteration % accumulate_iter == 0:
                # set nan grads to 0
                for param in model.parameters():
                    if param.grad is not None:
                        nan_to_num(
                            param.grad,
                            nan=0,
                            posinf=1e5,
                            neginf=-1e5,
                            out=param.grad,
                        )
                total_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_grad
                )
                storage.put_scalar(
                    "total_grad_norm", float(total_norm), smoothing_hint=False
                )
                optimizer.step()

            if iteration % accumulate_iter == 0:
                optimizer.zero_grad()
                storage.put_scalar(
                    "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
                )
                scheduler.step()
            # endregion

            # region tbx-recording
            # ------------------------------------------------------------------
            # recording losses&norm
            # ------------------------------------------------------------------
            # record losses
            if do_syn_sup:
                tbx_writer.add_scalar("SynLosses/SUM", float(losses.item()), iteration)
                for name, value in syn_loss_dict.items():
                    loss_name = "SelfLosses/{}".format(name)
                    loss_val = float(value)
                    tbx_writer.add_scalar(loss_name, loss_val, iteration)
            elif do_self:
                tbx_writer.add_scalar(
                    "SelfLosses/SUM", float(self_losses.item()), iteration
                )
                for name, value in self_loss_dict.items():
                    loss_name = "SelfLosses/{}".format(name)
                    loss_val = float(value)
                    tbx_writer.add_scalar(loss_name, loss_val, iteration)
                if do_weakly and this_iter_forward_mode == "pose":
                    tbx_writer.add_scalar(
                        "WeaklyLosses/SUM", float(weakly_losses.item()), iteration
                    )
                    for name, value in ws_loss_dict.items():
                        loss_name = "WeaklyLosses/{}".format(name)
                        loss_val = float(value)
                        tbx_writer.add_scalar(loss_name, loss_val, iteration)
            else:
                raise RuntimeError("Not in do_self or do_syn_sup")
            # record grad-norm
            tbx_writer.add_scalar("Grad/total_norm", float(total_norm), iteration)
            # endregion

            # region EMA
            # ------------------------------------------------------------------
            # update teacher model using ema
            # ------------------------------------------------------------------
            if (
                ema is not None
                and (iteration + 1) % (cfg.MODEL.EMA.UPDATE_FREQ * iters_per_epoch) == 0
            ):
                ema.update(model)
                ema.update_attr(model)
            # endregion

            # region periodically eval
            # ------------------------------------------------------------------
            # do test periodically or after training
            # ------------------------------------------------------------------
            if (
                eval_period > 0
                and (iteration + 1) % eval_period == 0
                and iteration != max_iter - 1
            ):
                eval_res = do_test(cfg, model, epoch=epoch, iteration=iteration)
                for title, test_result in eval_res.items():
                    tb_eval_markdown_str = eval_result_to_markdown(test_result)
                    tbx_writer.add_text(
                        "Eval metrics/{}".format(title), tb_eval_markdown_str, iteration
                    )

                torch.save(model.state_dict(), "student_{}.pth".format(epoch))
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            # endregion

            # region others
            # ------------------------------------------------------------------
            # some visualization
            # ------------------------------------------------------------------
            if is_log_iter:
                if len(writers) > 0:
                    writers["terminal"].write()
                    writers["tbx"].write()
                # visualize some images ========================================
                if cfg.TRAIN.VIS_IMG:
                    with torch.no_grad():
                        vis_i = 0
                        roi_img_vis = batch["roi_img"][vis_i].cpu().numpy()
                        roi_img_vis = (
                            denormalize_image(roi_img_vis, cfg)
                            .transpose(1, 2, 0)
                            .astype("uint8")
                        )
                        tbx_writer.add_image("input_image", roi_img_vis, iteration)

                        out_coor_x = out_dict["coor_x"].detach()
                        out_coor_y = out_dict["coor_y"].detach()
                        out_coor_z = out_dict["coor_z"].detach()
                        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)

                        out_xyz_vis = out_xyz[vis_i].cpu().numpy().transpose(1, 2, 0)
                        out_xyz_vis = get_emb_show(out_xyz_vis)
                        tbx_writer.add_image("out_xyz", out_xyz_vis, iteration)

                        gt_xyz_vis = (
                            batch["roi_xyz"][vis_i].cpu().numpy().transpose(1, 2, 0)
                        )
                        gt_xyz_vis = get_emb_show(gt_xyz_vis)
                        tbx_writer.add_image("gt_xyz", gt_xyz_vis, iteration)

                        out_mask = out_dict["mask"].detach()
                        out_mask = get_out_mask(cfg, out_mask)
                        out_mask_vis = out_mask[vis_i, 0].cpu().numpy()
                        tbx_writer.add_image("out_mask", out_mask_vis, iteration)

                        gt_mask_vis = batch["roi_mask"][vis_i].detach().cpu().numpy()
                        tbx_writer.add_image("gt_mask", gt_mask_vis, iteration)

            # ------------------------------------------------------------------
            # checkpointer step
            # ------------------------------------------------------------------
            periodic_checkpointer.step(iteration, epoch=epoch)
            # endregion

        if cfg.TRAIN.DEBUG_SINGLE_IM:
            mmcv.dump(
                debug_results,
                osp.join(
                    cfg.OUTPUT_DIR, "debug_results_{}.pkl".format(train_dset_names[0])
                ),
            )


def vis_train_data(data, obj_names, cfg):
    for i, d in enumerate(data):
        # if i >= 1:
        #     continue
        full_img = mmcv.imread(d["file_name"], "color")
        # if "000009/rgb/000047.png" not in d["file_name"]:
        #     continue
        print(d["file_name"])
        im_H, im_W = full_img.shape[:2]
        roi_cls = d["roi_cls"]
        if roi_cls not in [0]:
            continue
        bbox_center = d["bbox_center"]
        scale = d["scale"]
        x1 = max(min(bbox_center[0] - scale / 2, im_W), 0)
        x2 = max(min(bbox_center[0] + scale / 2, im_W), 0)
        y1 = max(min(bbox_center[1] - scale / 2, im_H), 0)
        y2 = max(min(bbox_center[1] + scale / 2, im_H), 0)
        full_img_vis = vis_bbox_opencv(full_img, np.array([x1, y1, x2, y2]), fmt="xyxy")

        bbox_ori = d["bbox"]
        full_img_bbox = vis_bbox_opencv(full_img, bbox_ori, fmt="xyxy")
        obj_name = obj_names[roi_cls]

        roi_img = d["roi_img"].numpy()
        roi_img = denormalize_image(roi_img, cfg).transpose(1, 2, 0).astype("uint8")

        roi_mask_trunc = d["roi_mask_trunc"].numpy().astype("bool")
        roi_mask_visib = d["roi_mask_visib"].numpy().astype("bool")
        roi_mask_obj = d["roi_mask_obj"].numpy().astype("bool")

        kernel = np.ones((3, 3), np.uint8)
        erode_mask_obj = cv2.erode(roi_mask_obj.astype("uint8"), kernel, iterations=1)

        roi_xyz = d["roi_xyz"].numpy().transpose(1, 2, 0)
        roi_xyz_show = get_emb_show(roi_xyz) * erode_mask_obj[:, :, None].astype(
            "float32"
        )

        coord2d = d["roi_coord_2d"].numpy().transpose(1, 2, 0)
        roi_h, roi_w = coord2d.shape[:2]
        zeros_1 = np.zeros((roi_h, roi_w, 1), dtype="float32")
        coord2d_3 = np.concatenate([zeros_1, get_emb_show(coord2d)], axis=2)

        # yapf: disable
        vis_imgs = [
            full_img_vis[:, :, [2, 1, 0]], full_img_bbox[:, :, [2, 1, 0]], roi_img[:, :, [2, 1, 0]],
            roi_mask_trunc * erode_mask_obj, roi_mask_visib * erode_mask_obj, roi_mask_obj * erode_mask_obj,
            roi_xyz_show,
            coord2d_3,
            coord2d[:, :, 0], coord2d[:, :, 1]
        ]
        titles = [
            "full_img", "ori_bbox", f"roi_img({obj_name})",
            "roi_mask_trunc", "roi_mask_visib", "roi_mask_obj",
            "roi_xyz",
            "roi_coord2d",
            "roi_coord2d_x", "roi_coord2d_y"
        ]
        row = 3
        col = 4
        if "roi_region" in d:
            roi_region = d["roi_region"].numpy()  # (bh, bw)
            roi_region_3 = np.zeros((roi_h, roi_w, 3), dtype="float32")
            for region_id in range(256):
                # if region_id == 0:
                #     continue
                if region_id in roi_region:
                    for _c in range(3):
                        roi_region_3[:, :, _c][roi_region == region_id] = roi_xyz_show[:, :, _c][
                            roi_region == region_id].mean()
            roi_region_3 = roi_region_3 * erode_mask_obj[:, :, None].astype("float32")
            vis_imgs.append(roi_region_3)
            titles.append("roi_region")
        if len(vis_imgs) > row * col:
            col += 1
        for _im, _name in zip(vis_imgs, titles):
            save_path = osp.join(cfg.OUTPUT_DIR, "vis", _name + '.png')
            mmcv.mkdir_or_exist(osp.dirname(save_path))
            if _im.shape[-1] == 3:
                _im = _im[:, :, [2, 1, 0]]
            if _im.max() < 1.1:
                _im = (_im * 255).astype("uint8")
            print(save_path)
            mmcv.imwrite(_im, save_path)

        grid_show(vis_imgs, titles, row=row, col=col)

        # yapf: enable


def eval_result_to_markdown(eval: str):
    data = eval.split("\n")
    num_col = len(data[0].split())
    res = "|" + "|".join(data[0].split()) + "|"
    res += "  \n|" + "|".join([" :----: "] * num_col) + "|"
    for row in data[1:]:
        res += "  \n|" + "|".join(row.split()) + "|"
    return res
