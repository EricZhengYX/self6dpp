import mmcv
import numpy as np

save_name = "pose_refine_ruida_lmo_NoBopTest_ape_train.json"
file_pth = "output/deepim/lmPbrSO/FlowNet512_1.5AugCosyAAEGray_Aggressive_Flat_lmPbr_SO/ape/inference_deepim_pbr/lmo_NoBopTest_ape_train/results.pkl"
detection_file = "datasets/BOP_DATASETS/lmo/test/init_poses/resnest50d_online_AugCosyAAEGray_mlBCE_DoubleMask_lmo_pbr_100e_so_withYolov4PbrBbox_wDeepimPbrPose_lmo_NoBopTest_train.json"
result = mmcv.load(file_pth)
detections = mmcv.load(detection_file)


def bbox_xyxy2xywh(bbox):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    return [x0, y0, w, h]


cvrt = {}
for img_name, img_dict in result.items():
    if img_name not in cvrt:
        cvrt[img_name] = []
    cur_list = cvrt[img_name]
    det = {
        single_det["obj_id"]: single_det
        for single_det in detections[img_name]
    }
    for obj_dict in img_dict:
        obj_id = obj_dict['obj_id']
        this_det = det[obj_id]
        cur_list.append({
            'bbox_est': this_det['bbox_est'],
            'obj_id': obj_id,
            'pose_est': obj_dict['pose_0'].tolist(),
            'pose_refine': obj_dict['pose_4'].tolist(),
            'score': this_det['score'],
        })
mmcv.dump(cvrt, save_name)
