import os
import os.path as osp
import re
import mmcv
import torch
import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from lib.dr_utils.dib_renderer_x.renderer_dibr import Renderer_dibr
from core.self6dpp.datasets.lm_dataset_d2 import LM_DICT, LM_13_OBJECTS, LM_OCC_OBJECTS
from lib.pysixd.inout import load_ply
from core.utils.data_utils import compute_vf

matplotlib.use("TkAgg")

cls_name = "ape"
pbr_cache_pth = (
    ".cache/dataset_dicts_lm_pbr_ape_train_beeefb9c0a880ca05b58607843b4b15b.pkl"
)
assert cls_name in pbr_cache_pth, "only support one cls every running now!"
if "lm_pbr" in pbr_cache_pth:
    dataset_type = "lm"
elif "lmo_pbr" in pbr_cache_pth:
    dataset_type = "lmo"
    raise NotImplementedError("not done")
else:
    raise NotImplementedError
vf_key = "vf_file"
DEBUG = False
vf_file_suffix = ".pkl"


def load_fps(pth="datasets/BOP_DATASETS/lm/models/fps_points.pkl", cls_name=cls_name, nfps=16):
    fps = mmcv.load(pth)[str(LM_DICT[cls_name])]
    key_name = 'fps{}_and_center'.format(nfps)
    assert key_name in fps, fps.keys()
    return fps[key_name][:-1]  # nfps*3


if __name__ == "__main__":
    pbr_cache = mmcv.load(pbr_cache_pth)
    fps16 = load_fps()

    for single_img_dict in tqdm(pbr_cache):
        if "annotations" not in single_img_dict:
            continue
        K = single_img_dict["cam"]
        H, W = single_img_dict["height"], single_img_dict["width"]
        fake_mask = np.ones((H, W))

        for single_obj_anno in single_img_dict["annotations"]:
            pose = single_obj_anno["pose"]
            if single_obj_anno.get('mask_full_file', None) is not None:
                vf_pth = (
                    single_obj_anno['mask_full_file']
                    .replace("mask", "vf")
                    .replace(".png", vf_file_suffix)
                )
            elif single_obj_anno.get('xyz_path', None) is not None:
                vf_pth = (
                    single_obj_anno['xyz_path']
                    .replace("xyz_crop", "vf")
                    .replace("-xyz.pkl", vf_file_suffix)
                )
            else:
                raise ValueError("Cannot phrase vf path.")
            single_obj_anno[vf_key] = vf_pth
            if osp.exists(vf_pth):
                continue
            mmcv.mkdir_or_exist(osp.dirname(vf_pth))
            vf, _ = compute_vf(fake_mask, fake_mask, fps16, K, pose)

            mmcv.dump(vf.astype(np.float16), vf_pth)  # approx 300k
