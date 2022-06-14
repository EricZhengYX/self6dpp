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
vf_file_suffix = ".png"
NFPS = 16
nchar = len(str(NFPS))


def load_fps(pth="datasets/BOP_DATASETS/lm/models/fps_points.pkl", cls_name=cls_name, nfps=16):
    fps = mmcv.load(pth)[str(LM_DICT[cls_name])]
    key_name = 'fps{}_and_center'.format(nfps)
    assert key_name in fps, fps.keys()
    return fps[key_name][:-1]  # nfps*3


def scale_0255(m):
    assert m.max() <= 1 and m.min() >= -1
    return ((m + 1) / 2 * 255).astype(np.uint8)


if __name__ == "__main__":
    pbr_cache = mmcv.load(pbr_cache_pth)
    fps16 = load_fps(nfps=NFPS)

    for single_img_dict in tqdm(pbr_cache):
        if "annotations" not in single_img_dict:
            continue
        K = single_img_dict["cam"]
        H, W = single_img_dict["height"], single_img_dict["width"]

        for single_obj_anno in single_img_dict["annotations"]:
            pose = single_obj_anno["pose"]
            if single_obj_anno.get('mask_full_file', None) is not None:
                _pth = single_obj_anno['mask_full_file']
                base_pth = osp.abspath(osp.join(_pth, "../../.."))
                seq_name = _pth.split('/')[-3]
                obj_name = re.match("\d+_\d+", _pth.split('/')[-1]).group()
            elif single_obj_anno.get('xyz_path', None) is not None:
                _pth = single_obj_anno['xyz_path']
                base_pth = osp.abspath(osp.join(_pth, "../../.."))
                seq_name = _pth.split('/')[-2]
                obj_name = re.match("\d+_\d+", _pth.split('/')[-1]).group()
            else:
                raise ValueError("Cannot phrase vf path.")
            vf_full_pth = osp.join(base_pth, "vf_full", seq_name, obj_name)
            vf_visib_pth = osp.join(base_pth, "vf_visib", seq_name, obj_name)
            mask_full_pth = osp.join(base_pth, seq_name, "mask", obj_name+".png")
            mask_visib_pth = osp.join(base_pth, seq_name, "mask_visib", obj_name+".png")

            single_obj_anno["vf_full_base"] = vf_full_pth
            single_obj_anno["vf_visib_base"] = vf_visib_pth
            # currently cannot resume!
            mmcv.mkdir_or_exist(vf_full_pth)
            mmcv.mkdir_or_exist(vf_visib_pth)

            # load masks
            mask_full = np.asarray(Image.open(mask_full_pth)) > 0
            mask_visib = np.asarray(Image.open(mask_visib_pth)) > 0

            vf_f, vf_v = compute_vf(mask_full, mask_visib, fps16, K, pose)
            vf_f_255 = scale_0255(vf_f)
            vf_v_255 = scale_0255(vf_v)

            # for vf full
            for fps_i in range(NFPS):
                for ci, ch in enumerate(["u", "v"]):
                    img = vf_f_255[..., fps_i, ci]
                    save_pth = osp.join(vf_full_pth, str(fps_i).zfill(nchar) + ch + ".png")
                    Image.fromarray(img).save(save_pth)

            # for vf visib
            for fps_i in range(NFPS):
                for ci, ch in enumerate(["u", "v"]):
                    img = vf_v_255[..., fps_i, ci]
                    save_pth = osp.join(vf_visib_pth, str(fps_i).zfill(nchar) + ch + ".png")
                    Image.fromarray(img).save(save_pth)
    mmcv.dump(pbr_cache, osp.basename(pbr_cache_pth))
