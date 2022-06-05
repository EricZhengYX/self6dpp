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

matplotlib.use("TkAgg")

pbr_cache_pth = (
    ".cache/dataset_dicts_lm_pbr_13_train_beeefb9c0a880ca05b58607843b4b15b.pkl"
)
if "lm_pbr" in pbr_cache_pth:
    dataset_type = "lm"
elif "lmo_pbr" in pbr_cache_pth:
    dataset_type = "lmo"
    raise NotImplementedError("not done")
else:
    raise NotImplementedError
norm_key = "norm_file"
DEBUG = False
AS_IMAGE = True


def load_models(pth="datasets/BOP_DATASETS/lm/models"):
    preloaded = "models_all_w_name.pkl"
    full_pth = osp.abspath(osp.join(pth, preloaded))
    if osp.exists(full_pth):
        return mmcv.load(full_pth)
    ply_list = sorted(
        [
            osp.abspath(osp.join(pth, file))
            for file in os.listdir(pth)
            if osp.splitext(file)[-1] == ".ply"
        ]
    )
    models = {}
    for ply_pth in ply_list:
        obj_id = int(re.search("\d+", osp.basename(ply_pth)).group())
        obj_name = LM_DICT[obj_id]
        models[obj_name] = load_ply(ply_pth, vertex_scale=0.001)
    mmcv.dump(models, full_pth)
    return models


if __name__ == "__main__":
    pbr_cache = mmcv.load(pbr_cache_pth)
    models = load_models()

    renderer = Renderer_dibr(480, 640, "VertexColorBatch")

    for single_img_dict in tqdm(pbr_cache):
        if "annotations" not in single_img_dict:
            continue
        K = single_img_dict["cam"]
        H, W = single_img_dict["height"], single_img_dict["width"]
        for single_obj_anno in single_img_dict["annotations"]:
            if norm_key in single_obj_anno and single_obj_anno[norm_key] is not None:
                continue
            norm_pth = (
                single_obj_anno["mask_full_file"]
                .replace("mask", "norm")
                .replace(".png", ".pkl")
            )
            mmcv.mkdir_or_exist(osp.dirname(norm_pth))
            category_id = single_obj_anno["category_id"]
            cate_name = LM_13_OBJECTS[
                category_id
            ]  # TODO: maybe different when using LMO?

            if DEBUG:
                x, y, h, w = single_obj_anno["bbox"]
                cx, cy = x + h / 2, y + h / 2
                plt.imshow(Image.open(single_img_dict["file_name"]))
                plt.annotate(str(category_id), xy=[cx, cy], size=20, color="r")
                plt.show()

            # poses
            pose = torch.tensor(single_obj_anno["pose"]).cuda()
            std_dtype = pose.dtype
            R, t = pose[None, :3, :3], pose[None, :3, 3:]

            # models
            model = models[cate_name]
            model_to_render = {
                "vertices": torch.tensor(model["pts"],     device='cuda', dtype=std_dtype),
                "colors":   torch.tensor(model["colors"],  device='cuda', dtype=std_dtype),
                "normals":  torch.tensor(model["normals"], device='cuda', dtype=std_dtype),
                "faces":    torch.tensor(model["faces"],   device='cuda', dtype=std_dtype),
            }

            # do rendering and save
            renderings = renderer.render_batch(
                R, t, [model_to_render], Ks=K, width=W, height=H, mode=["norm"]
            )
            norm_fig = renderings["norm"][0].cpu().numpy()
            if AS_IMAGE:
                Image.fromarray((norm_fig * 255).astype(np.uint8)).save(
                    norm_fig
                )  # approx 11.4k
            else:
                # saving as image(0-255) may bring round-off error
                mmcv.dump(norm_fig, norm_pth)  # approx 3.7M
