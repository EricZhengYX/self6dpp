import mmcv
import numpy as np

save_name = "pose_refine_extra0043474_lmo_NoBopTest_ape_train.json"
file_pth = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/ape/inference_deepim_pbr/lmo_NoBopTest_ape_train/results.pkl"
result = mmcv.load(file_pth)


def bbox_xyxy2xywh(bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return [x0, y0, w, h]


cvrt = {}
for img_name, img_dict in result.items():
    if img_name not in cvrt:
        cvrt[img_name] = []
    cur_list = cvrt[img_name]
    for obj_dict in img_dict:
        cur_list.append({
            'bbox_est': bbox_xyxy2xywh(obj_dict['bbox_det_xyxy']),
            'obj_id': obj_dict['obj_id'],
            'pose_est': obj_dict['pose_0'].tolist(),
            'pose_refine': obj_dict['pose_4'].tolist(),
            'score': obj_dict['score'],
        })
mmcv.dump(cvrt, save_name)
